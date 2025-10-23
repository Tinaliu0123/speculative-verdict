"""
Computes global consensus scores between models based on cross/self perplexity scores.
"""

from __future__ import annotations

import json
import math
import argparse
import random
import logging
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class RankingConfig:
    """Configuration for consensus ranking computation."""
    top_k: int = 3
    eps: float = 1e-12 
    seed: Optional[int] = None 
    
    def __post_init__(self):
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.eps <= 0:
            raise ValueError("eps must be positive")

def load_entries(input_path: Union[str, Path]) -> Iterable[Dict[str, Any]]:
    """Load entries from JSON array or JSONL file."""
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with input_path.open('r', encoding='utf-8') as f:
        pos = f.tell()
        first_char = None
        
        while True:
            char = f.read(1)
            if not char:
                break
            if not char.isspace():
                first_char = char
                break
        
        f.seek(pos)
        
        if first_char == '[':
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Top-level JSON must be an array")
            for item in data:
                if isinstance(item, dict):
                    yield item
        else:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

def save_results(results: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    """Save results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.write('\n')
    logging.info(f"Results saved to {output_path}")

def assign_dense_ranks(items: List[Dict[str, Any]], key_func: callable, eps: float = 1e-12) -> None:
    """Assign dense ranks to items based on key function (modifies in-place)."""
    current_rank = 0
    prev_value: Optional[float] = None
    
    for item in items:
        value = key_func(item)
        
        if value is None:
            item["rank"] = None
            continue
            
        if prev_value is None or abs(value - prev_value) > eps:
            current_rank += 1
            prev_value = value
            
        item["rank"] = current_rank

def select_topk_with_ties(
    sorted_items: List[Dict[str, Any]],
    k: int,
    key_func: callable,
    rng: random.Random,
    eps: float = 1e-12
) -> List[Dict[str, Any]]:
    """Select top-k items, handling ties with random sampling."""
    if not sorted_items or k <= 0:
        return []
        
    selected = []
    i = 0
    n = len(sorted_items)
    
    while i < n and len(selected) < k:
        current_value = key_func(sorted_items[i])
        
        if current_value is None:
            break 
            
        # Find tie group
        j = i + 1
        while j < n:
            next_value = key_func(sorted_items[j])
            if next_value is None or abs(next_value - current_value) > eps:
                break
            j += 1
            
        tie_group = sorted_items[i:j]
        remaining_slots = k - len(selected)
        
        if len(tie_group) <= remaining_slots:
            selected.extend(tie_group)
        else:
            selected.extend(rng.sample(tie_group, remaining_slots))
            
        i = j
        
    return selected

def safe_log(value: Optional[Union[int, float]]) -> Optional[float]:
    """Compute log of value if positive, otherwise return None."""
    if isinstance(value, (int, float)) and value > 0:
        return math.log(value)
    return None

def compute_consensus_ranking(entry: Dict[str, Any], config: RankingConfig) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Compute consensus ranking directly from cross/self PPL.
    
    For each model B: aggregate_score(B) = Σ|d(A→B)| for all A≠B
    where d(A→B) = log(cross_ppl[A][B]) - log(self_ppl[B])
    
    Returns:
        (all_scores, top_k)
    """
    self_ppl = entry.get("self_ppl") or {}
    cross_ppl = entry.get("cross_ppl") or {}
    
    if not self_ppl or not cross_ppl:
        raise ValueError(f"Missing PPL data")
    
    models = list(self_ppl.keys())
    
    # Precompute log(self_ppl)
    log_self = {b: safe_log(self_ppl.get(b)) for b in models}

    # Compute pairwise details for consensus_ranks output
    consensus_ranks = {}
    for model_a in models:
        pairwise_scores = []
        for model_b in models:
            if model_a == model_b:
                continue
            
            log_cross = safe_log(cross_ppl.get(model_a, {}).get(model_b))
            log_self_b = log_self.get(model_b)
            
            if log_cross is not None and log_self_b is not None:
                d_value = log_cross - log_self_b
                abs_d_value = abs(d_value)
            else:
                d_value = None
                abs_d_value = None
            
            pairwise_scores.append({
                "model": model_b,
                "d": d_value,
                "abs_d": abs_d_value
            })
        
        pairwise_scores.sort(key=lambda x: float('inf') if x["abs_d"] is None else x["abs_d"])
        assign_dense_ranks(pairwise_scores, key_func=lambda x: x["abs_d"], eps=config.eps)
        consensus_ranks[model_a] = pairwise_scores
    
    # Compute aggregate score
    all_scores = []
    for model_a in models:
        abs_d_values = []
        for model_b in models:
            if model_a == model_b:
                continue
            log_cross = safe_log(cross_ppl.get(model_a, {}).get(model_b))
            log_self_b = log_self.get(model_b)
            if log_cross is not None and log_self_b is not None:
                abs_d_values.append(abs(log_cross - log_self_b))
        
        score_sum = sum(abs_d_values) if abs_d_values else None
        all_scores.append({
            "model": model_a, 
            "score_sum": score_sum,
            "n_pairs": len(abs_d_values),
            "rank": None
        })
    
    # Sort by score_sum ascending (lower is better)
    all_scores.sort(key=lambda x: float('inf') if x["score_sum"] is None else x["score_sum"])
    assign_dense_ranks(all_scores, key_func=lambda x: x["score_sum"], eps=config.eps)
    
    # Select top-k
    rng = random.Random(config.seed)
    valid_scores = [s for s in all_scores if s["score_sum"] is not None]
    actual_top_k = min(config.top_k, len(valid_scores))
    
    top_k = select_topk_with_ties(all_scores, actual_top_k, 
                                   key_func=lambda x: x["score_sum"], rng=rng, eps=config.eps)

    return consensus_ranks, all_scores, top_k


def process_entry(entry: Dict[str, Any], entry_idx: int, config: RankingConfig) -> Dict[str, Any]:
    """Process a single entry to compute consensus rankings."""
    try:
        self_ppl = entry.get("self_ppl") or {}
        models = list(self_ppl.keys())
        
        if len(models) < 2:
            raise ValueError(f"Need at least 2 models, got {len(models)}")
        
        consensus_ranks, all_scores, top_k = compute_consensus_ranking(entry, config)
        
        return {
            "idx": entry_idx,
            "question": entry.get("question"),
            "image_path": entry.get("image_path"),
            "dataset": entry.get("dataset"),
            "n_models": len(models),
            "consensus_ranks": consensus_ranks,
            "final_ranking": all_scores,
            "final_topk": top_k,
            "config": {
                "top_k": config.top_k,
                "eps": config.eps,
                "seed": config.seed
            }
        }
        
    except Exception as e:
        logging.error(f"Entry {entry_idx}: {type(e).__name__}: {e}")
        return {
            "idx": entry_idx,
            "question": entry.get("question"),
            "image_path": entry.get("image_path"),
            "error": f"{type(e).__name__}: {e}"
        }

def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute model consensus rankings from cross/self perplexity scores"
    )
    
    parser.add_argument("--input", required=True, help="Input file (JSON/JSONL)")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top models (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--eps", type=float, default=1e-12, help="Tie threshold (default: 1e-12)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    return parser.parse_args()

def main() -> None:
    """Main entry point."""
    args = parse_arguments()
    setup_logging(args.verbose)
    
    config = RankingConfig(
        top_k=args.top_k,
        eps=args.eps,
        seed=args.seed
    )
    
    logging.info(f"Starting consensus ranking: {config}")
    logging.info(f"Input: {args.input}, Output: {args.output}")
    
    results = []
    entry_count = 0
    error_count = 0
    
    for entry_idx, entry in enumerate(load_entries(args.input)):
        result = process_entry(entry, entry_idx, config)
        results.append(result)
        
        entry_count += 1
        if "error" in result:
            error_count += 1
            
        if entry_count % 100 == 0:
            logging.info(f"Processed {entry_count} entries ({error_count} errors)")
    
    save_results(results, args.output)
    
    logging.info(f"Completed: {entry_count} entries, {error_count} errors")

if __name__ == "__main__":
    main()