"""
Entry point for the draft & verdict stage.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import random
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional
from PIL import Image

import numpy as np
import torch
from qwen_vl_utils import process_vision_info

from draft import (
    QAEntry,
    ReasoningResult,
    PrefillResult,
    run_inference,
    run_prefill,
    prepare_models,
    load_vlm
)
from verdict import qwen_verdict, gpt4o_verdict

from prompts import detect_dataset_from_path, get_legacy_prompts
from utils.post_process import extract_final_boxed_content, clean_think_tags, clean_answer

DEFAULT_FLUSH_EVERY = 10

def setup_logging(verbosity: int = 1) -> None:
    """Setup concise root logger."""
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")

def set_all_seeds(seed: Optional[int]) -> None:
    """For reproducibility."""
    if seed is None:
        logging.info("No seed provided (non-deterministic run).")
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Seeds set to {seed}.")

def iter_entries_auto(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    """Stream entries from JSON array or JSONL without loading entire file into memory."""
    with path.open("r", encoding="utf-8") as f:
        pos = f.tell()
        first = None
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first = ch
                break
        f.seek(pos)

        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Top-level JSON must be an array.")
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj

def safe_write_json(obj: Any, path: pathlib.Path) -> None:
    """Pretty-print JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fo:
        json.dump(obj, fo, ensure_ascii=False, indent=2)
        fo.write("\n")

def to_QAEntry(d: Dict[str, Any], dataset: Optional[str] = None) -> QAEntry:
    """Normalize raw dict from input file into QAEntry."""
    image_path = d.get("image_path")
    question = d.get("question")
    answers = d.get("answers") or d.get("ground_truths")
    extra = {k: v for k, v in d.items() if k not in {"image_path", "question", "answers", "ground_truths"}}
    
    if not isinstance(image_path, str) or not isinstance(question, str):
        raise ValueError("Each entry must contain 'image_path' (str) and 'question' (str).")
    if answers is not None:
        if isinstance(answers, str):
            answers = [answers]
        elif not isinstance(answers, list):
            raise ValueError(f"'answers' must be a list or string, got {type(answers).__name__}.")
        
    return QAEntry(image_path=image_path, question=question, answers=answers, extra=extra)

def run_inference_batch(
    in_path: pathlib.Path,
    out_path: pathlib.Path,
    *,
    model_paths: List[str],
    inference_mode: str = "reason",  # "qa" or "reason"
    dataset: Optional[str] = None,
    start_idx: int = 0,
    max_entries: Optional[int] = None,
    flush_every: int = DEFAULT_FLUSH_EVERY,
    merge_output: bool = False,
) -> None:
    """Execute per-model generative reasoning over a dataset."""
    models = prepare_models(model_paths)

    # Load existing results if merge mode
    existing_results = {}
    if merge_output and out_path.exists():
        logging.info(f"Merge mode: loading existing results from {out_path}")
        with out_path.open('r') as f:
            existing_data = json.load(f)
        # Index by (image_path, question) for fast lookup
        for item in existing_data:
            key = (item["image_path"], item["question"])
            existing_results[key] = item

    results: List[Dict[str, Any]] = []

    processed = 0
    for idx, raw in enumerate(iter_entries_auto(in_path)):
        if idx < start_idx:
            continue
        if max_entries is not None and processed >= max_entries:
            break

        entry = to_QAEntry(raw, dataset)
        key = (entry.image_path, entry.question)
        out: ReasoningResult = run_inference(
            models, 
            entry, 
            inference_mode=inference_mode, 
            q_idx=idx + 1, 
            dataset=dataset
        )

        if key in existing_results:
            existing_item = existing_results[key]
            # Merge models_reasoning dictionaries
            existing_item["models_reasoning"].update(asdict(out)["models_reasoning"])
            results.append(existing_item)
            logging.info(f"Merged results for entry {idx+1}")
        else:
            results.append(asdict(out))

        processed += 1
        if processed % flush_every == 0:
            logging.info(f"[Reasoning-{dataset}] Flushing @ {processed} → {out_path}")
            safe_write_json(results, out_path)
        torch.cuda.empty_cache()

    safe_write_json(results, out_path)
    logging.info(f"[Reasoning-{dataset}] Completed {processed} examples. Output: {out_path}")

def run_prefill_batch(
    in_path: pathlib.Path,
    out_path: pathlib.Path,
    *,
    model_paths: List[str],
    mode: str = "cross",
    source_key: str = "models_reasoning",
    running_model: Optional[str] = None,
    dataset: Optional[str] = None,
    start_idx: int = 0,
    max_entries: Optional[int] = None,
    flush_every: int = DEFAULT_FLUSH_EVERY,
    merge_output: bool = False,
) -> None:
    """Execute prefill scoring (self/cross PPL) over a dataset."""
    models = prepare_models(model_paths)

    # Load existing results if merge mode
    existing_results = {}
    if merge_output and out_path.exists():
        logging.info(f"Merge mode: loading existing results from {out_path}")
        try:
            with out_path.open('r') as f:
                existing_data = json.load(f)
            # Index by (image_path, question)
            for item in existing_data:
                key = (item["image_path"], item["question"])
                existing_results[key] = item
            logging.info(f"Loaded {len(existing_results)} existing entries")
        except Exception as e:
            logging.warning(f"Could not load existing results: {e}")

    results: List[Dict[str, Any]] = []

    processed = 0
    for idx, raw in enumerate(iter_entries_auto(in_path)):
        if idx < start_idx:
            continue
        if max_entries is not None and processed >= max_entries:
            break

        entry = to_QAEntry(raw)
        key = (entry.image_path, entry.question)
        
        out: PrefillResult = run_prefill(
            models, entry, mode="cross", source=source_key, 
            running_model=running_model, dataset=dataset
        )
        
        result_dict = asdict(out)
        
        # Merge with existing results if present
        if key in existing_results:
            existing_item = existing_results[key]
            
            # Merge self_ppl
            existing_item.setdefault("self_ppl", {}).update(result_dict["self_ppl"])
            
            # Merge cross_ppl
            if result_dict.get("cross_ppl"):
                existing_cross = existing_item.setdefault("cross_ppl", {})
                for answer_src, scores in result_dict["cross_ppl"].items():
                    existing_cross.setdefault(answer_src, {}).update(scores)
            
            # Update models list
            existing_models = set(existing_item.get("models", []))
            new_models = set(result_dict["models"])
            existing_item["models"] = list(existing_models | new_models)
            
            results.append(existing_item)
            logging.info(f"Merged prefill results for entry {idx+1}")
        else:
            results.append(result_dict)

        processed += 1
        if processed % flush_every == 0:
            logging.info(f"[Prefill-{mode}-{dataset}] Flushing @ {processed} → {out_path}")
            safe_write_json(results, out_path)
        torch.cuda.empty_cache()

    safe_write_json(results, out_path)
    logging.info(f"[Prefill-{mode}-{dataset}] Completed {processed} examples. Output: {out_path}")

def run_inference_from_topk_batch(
    in_path: pathlib.Path,
    out_path: pathlib.Path,
    consensus_path: pathlib.Path,
    *,
    model_tag_to_path: Dict[str, str], 
    inference_mode: str = "reason",
    dataset: Optional[str] = None,
    start_idx: int = 0,
    max_entries: Optional[int] = None,
    flush_every: int = DEFAULT_FLUSH_EVERY,
    merge_output: bool = False,
) -> None:
    """Execute inference only for top-k models selected by consensus scoring."""
    
    with consensus_path.open('r') as f:
        consensus_results = json.load(f)
    
    consensus_map = {
        (item["image_path"], item["question"]): [m["model"] for m in item.get("final_topk", [])]
        for item in consensus_results
    }
    logging.info(f"Loaded consensus data for {len(consensus_map)} entries")
    
    all_needed_tags = set()
    for topk_tags in consensus_map.values():
        all_needed_tags.update(topk_tags)

    logging.info(f"Models needed: {all_needed_tags}")
    
    needed_paths = [model_tag_to_path[tag] for tag in all_needed_tags if tag in model_tag_to_path]
    logging.info(f"Loading {len(needed_paths)} models...")
    
    all_models = prepare_models(needed_paths)
    tag_to_model = {model.tag: model for model in all_models}
    
    existing_results = {}
    if merge_output and out_path.exists():
        with out_path.open('r') as f:
            for item in json.load(f):
                key = (item["image_path"], item["question"])
                existing_results[key] = item
        logging.info(f"Loaded {len(existing_results)} existing entries")
    
    results = []
    processed = 0
    
    for idx, raw in enumerate(iter_entries_auto(in_path)):
        if idx < start_idx:
            continue
        if max_entries is not None and processed >= max_entries:
            break
        
        entry = to_QAEntry(raw)
        key = (entry.image_path, entry.question)
        
        topk_tags = consensus_map.get(key, [])
        if not topk_tags:
            logging.warning(f"Entry {idx}: No topk models, skipping")
            continue
        
        entry_models = [tag_to_model[tag] for tag in topk_tags if tag in tag_to_model]
        if not entry_models:
            logging.warning(f"Entry {idx}: No valid models for {topk_tags}, skipping")
            continue
        
        out: ReasoningResult = run_inference(
            entry_models, entry, 
            inference_mode=inference_mode, 
            q_idx=idx + 1, 
            dataset=dataset
        )
        
        result_dict = asdict(out)
        
        # Merge if needed
        if key in existing_results:
            existing_results[key]["models_reasoning"].update(result_dict["models_reasoning"])
            results.append(existing_results[key])
        else:
            results.append(result_dict)
        
        processed += 1
        if processed % flush_every == 0:
            logging.info(f"[Inference-topk] Processed {processed}/{len(consensus_map)}")
            safe_write_json(results, out_path)
        
        torch.cuda.empty_cache()
    
    safe_write_json(results, out_path)
    logging.info(f"Completed {processed} entries using {len(all_models)} models")

def run_verdict_batch(
    in_path: pathlib.Path,
    out_path: pathlib.Path,
    *,
    verdict_model_path: str,
    annotated_folder: str = "",
    dataset: Optional[str] = None,
    start_idx: int = 0,
    max_entries: Optional[int] = None,
    flush_every: int = DEFAULT_FLUSH_EVERY,
) -> None:
    """Execute verdict over a dataset with existing model reasoning."""
    
    model, processor, tokenizer, tag = load_vlm(verdict_model_path)
    results: List[Dict[str, Any]] = []

    processed = 0
    for idx, raw in enumerate(iter_entries_auto(in_path)):
        if idx < start_idx:
            continue
        if max_entries is not None and processed >= max_entries:
            break

        entry_dict = dict(raw)
        
        # Ensure required fields exist
        if "models_reasoning" not in entry_dict:
            raise ValueError(f"Entry {idx} missing 'models_reasoning' field required for verdict")

        full_response, final_answer = qwen_verdict(
            model=model,
            processor=processor,
            question=entry_dict["question"],
            answers_dict=entry_dict["models_reasoning"],
            orig_img_path=entry_dict["image_path"],
            annotated_folder=annotated_folder,
            dataset=dataset, 
            device="cuda"
        )

        entry_dict["final_reasoning"] = full_response
        entry_dict["final_answer"] = final_answer
        entry_dict["dataset"] = dataset

        results.append(entry_dict)

        processed += 1
        if processed % flush_every == 0:
            logging.info(f"[verdict-{dataset}] Flushing @ {processed} → {out_path}")
            safe_write_json(results, out_path)
        torch.cuda.empty_cache()

    safe_write_json(results, out_path)
    logging.info(f"[verdict-{dataset}] Completed {processed} examples. Output: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the 'Draft' stage with dataset-aware prompts: (1) reasoning, (2) prefill scoring, (3) verdict."
    )
    p.add_argument("--in_json", required=True, help="Input file (JSON array or JSONL).")
    p.add_argument("--out_json", required=True, help="Output JSON path.")
    p.add_argument("--consensus_file", default=None,
                   help="[inference_from_topk] Consensus scoring results with final_topk")
    p.add_argument("--model_mapping", type=str, default=None,
                   help='[inference_from_topk] JSON mapping of tag->path, e.g. \'{"qwen":"/path/to/qwen"}\'')
    p.add_argument("--mode",
                   choices=["inference", "inference_from_topk", "prefill_cross", "verdict"],  # 添加新模式
                   default="inference",
                   help="Pipeline stage to run.")
    p.add_argument("--models", nargs="+", required=False,
                   help="Model paths (space-separated). Required for inference/prefill modes.")
    
    # Dataset specification
    p.add_argument("--dataset", choices=["infovqa", "hrbench", "museum", "pro"], default=None,
                   help="Dataset type (auto-detected if not specified).")
    
    # Mode-specific parameters
    p.add_argument("--inference_mode", choices=["qa", "reason"], default="reason", 
                   help="[inference] Inference type: 'qa' for direct QA, 'reason' for reasoning.")
    p.add_argument("--source_key", default="models_reasoning",
                   help="[prefill_cross] key where per-model answers live.")
    p.add_argument("--running_model", default=None,
                   help="[prefill_cross] restrict cross-eval target.")
    p.add_argument("--annotated_folder", default="",
                   help="[verdict] folder containing layout-annotated images.")
    
    # Processing parameters
    p.add_argument("--start_idx", type=int, default=0,
                   help="Start from this 0-based index in the dataset.")
    p.add_argument("--max_entries", type=int, default=None,
                   help="Process at most N entries.")
    p.add_argument("--flush_every", type=int, default=DEFAULT_FLUSH_EVERY,
                   help="Flush partial results every N entries.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed (sets torch/cuda/numpy/python).")
    p.add_argument("-v", "--verbose", action="count", default=1,
                   help="Increase logging verbosity (-v: INFO, -vv: DEBUG).")
    p.add_argument("--merge_output", action="store_true",
                   help="Merge results into existing output file instead of overwriting")
                   
    return p.parse_args()

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    set_all_seeds(args.seed)

    in_path = pathlib.Path(args.in_json)
    out_path = pathlib.Path(args.out_json)
    model_paths = args.models

    if args.mode == "inference": 
        run_inference_batch( 
            in_path=in_path,
            out_path=out_path,
            model_paths=model_paths,
            inference_mode=args.inference_mode, 
            dataset=args.dataset,
            start_idx=args.start_idx,
            max_entries=args.max_entries,
            flush_every=args.flush_every,
            merge_output=args.merge_output
        )
    elif args.mode == "prefill_cross":
        run_prefill_batch(
            in_path=in_path,
            out_path=out_path,
            model_paths=model_paths,
            mode="cross",
            source_key=args.source_key,
            running_model=args.running_model,
            dataset=args.dataset,
            start_idx=args.start_idx,
            max_entries=args.max_entries,
            flush_every=args.flush_every,
            merge_output=args.merge_output,
        )
    elif args.mode == "inference_from_topk":
        if not args.consensus_file:
            raise ValueError("--consensus_file required for inference_from_topk mode")
        if not args.model_mapping:
            raise ValueError("--model_mapping required for inference_from_topk mode")
        
        import json
        model_tag_to_path = json.loads(args.model_mapping)
        
        run_inference_from_topk_batch(
            in_path=in_path,
            out_path=out_path,
            consensus_path=pathlib.Path(args.consensus_file),
            model_tag_to_path=model_tag_to_path, 
            inference_mode=args.inference_mode,
            dataset=args.dataset,
            start_idx=args.start_idx,
            max_entries=args.max_entries,
            flush_every=args.flush_every,
            merge_output=args.merge_output,
        )
    elif args.mode == "verdict":
        if not model_paths:
            raise ValueError("verdict mode requires at least one model path (the verdict model)")
        run_verdict_batch(
            in_path=in_path,
            out_path=out_path,
            verdict_model_path=model_paths[0],  # Use first model as verdict model
            annotated_folder=args.annotated_folder,
            dataset=args.dataset,
            start_idx=args.start_idx,
            max_entries=args.max_entries,
            flush_every=args.flush_every,
        )
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

if __name__ == "__main__":
    main()