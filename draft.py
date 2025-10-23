"""
Draft stage:
1. Generative inference across multiple VLM models
2. Cross-all consensus scoring

Core functions:
    run_inference: Generates draft answers from each model
    run_prefill: Computes consensus scores between model outputs
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from PIL import Image
import re
import json
import torch
import time

from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq
from transformers.image_utils import load_image

try:
    from transformers import Glm4vForConditionalGeneration
except ImportError:
    Glm4vForConditionalGeneration = None

try:
    from transformers import Gemma3ForConditionalGeneration
except ImportError:
    Gemma3ForConditionalGeneration = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoModelForImageTextToText, LlavaOnevisionForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration, AutoModelForImageTextToText, LlavaOnevisionForConditionalGeneration = None, None, None

# from evaluate import compute_anls
from utils.post_process import (
    extract_final_boxed_content,
    clean_think_tags,
    clean_answer,
    glm_extract_boxed,
    is_only_think_fragment,
    chartmuseum_extract_answer
)
from prompts import get_prompt, detect_dataset_from_path

@dataclass
class QAEntry:
    """Input entry for pipeline processing."""
    image_path: str
    question: str
    answers: Optional[List[str]] = None
    extra: Optional[Dict[str, Any]] = None

@dataclass
class PerModelRecord:
    """Per-model output for generative reasoning."""
    reasoning: str
    answer: str
    # anls: float

@dataclass
class ReasoningResult:
    """Output of run_inference: per-model reasoning and answers."""
    image_path: str
    question: str
    models_reasoning: Dict[str, PerModelRecord]
    final_reasoning: str = "NO_VERDICT"
    final_answer: str = "NO_VERDICT"
    # anls: float = 0.0
    ground_truths: Optional[List[str]] = None

@dataclass
class PrefillResult:
    """Output of run_prefill: self and cross prefill scores."""
    image_path: str
    question: str
    models: List[str]
    answers: Dict[str, Dict[str, Any]]
    self_ppl: Dict[str, float]
    cross_ppl: Optional[Dict[str, Dict[str, float]]] = None

# Model Loading
def load_vlm(model_path: str, device="cuda", dtype="bfloat16"):
    """Load VLM and return (model, processor, tokenizer, tag)"""
    if "qwen2.5-vl-7b" in model_path.lower() or "qwen2.5-vl-72b" in model_path.lower() :
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=getattr(torch, dtype), device_map="auto",
            low_cpu_mem_usage=True, attn_implementation="flash_attention_2"
        ).eval()
        MIN_PIXELS = 1280 * 28 * 28
        MAX_PIXELS = 16384 * 28 * 28
        proc = AutoProcessor.from_pretrained(model_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = "left" 
        proc.tokenizer = tokenizer
        return model, proc, tokenizer, "qwen"
    
    elif "qwen2.5-vl-3b" in model_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=getattr(torch, dtype), device_map="auto", attn_implementation="flash_attention_2"
        ).eval()
        MIN_PIXELS = 1280 * 28 * 28
        MAX_PIXELS = 16384 * 28 * 28
        proc = AutoProcessor.from_pretrained(model_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = "left" 
        proc.tokenizer = tokenizer
        return model, proc, tokenizer, "qwen"

    elif "glm" in model_path.lower():
        processor = AutoProcessor.from_pretrained(model_path)
        model = Glm4vForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        return model, processor, processor.tokenizer, "glm"

    elif "ovis" in model_path.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).cuda()
        tokenizer = model.text_tokenizer
        model.eval()
        return model, None, tokenizer, "ovis"

    elif "mimo" in model_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=getattr(torch, dtype), device_map="auto",
            low_cpu_mem_usage=True, attn_implementation="flash_attention_2"
        ).eval()
        proc = AutoProcessor.from_pretrained(model_path, min_pixels=0, max_pixels=4096*28*28)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        proc.tokenizer = tokenizer
        return model, proc, tokenizer, "mimo"

    elif "intern" in model_path.lower():
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, device_map=device, torch_dtype=torch.bfloat16
        )
        if "3.5" in model_path.lower():
            return model, processor, processor.tokenizer, "internvl3.5"
        return model, processor, processor.tokenizer, "internvl"

    elif "llava" in model_path.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=getattr(torch, dtype), device_map="auto", trust_remote_code=True
            )
        processor = AutoProcessor.from_pretrained(model_path, min_pixels = 256 * 28 * 28, max_pixels = 3240000, trust_remote_code=True)
        return model, processor, processor.tokenizer, "llava"

    elif "gemma" in model_path.lower():
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path, device_map="auto"
        ).eval()
        processor = AutoProcessor.from_pretrained(model_path)
        return model, processor, processor.tokenizer, "gemma"

    elif "eagle" in model_path.lower():
        model = AutoModel.from_pretrained(model_path,trust_remote_code=True, torch_dtype=torch.bfloat16)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        processor.tokenizer.padding_side = "left"
        return model, processor, processor.tokenizer, "eagle"

    else:
        raise ValueError(f"Unsupported model path: {model_path}")

def prepare_models(model_paths):
    """Prepare prefill models"""
    from model import PrefillingModel
    models = []
    for path in model_paths:
        m, p, tok, tag = load_vlm(path)
        models.append(PrefillingModel(m, p, tok, tag))
    return models

def _post_process_raw_response(raw_text: str, tag: str, dataset: Optional[str] = None, inference_mode: Optional[str] = None) -> str:
    """Post-process model response to extract final answer"""
    if not raw_text:
        return ""

    if dataset and any(k in dataset.lower() for k in ["museum","pro"]):
        if dataset == "pro" and inference_mode == "qa":
            return raw_text
        ans = chartmuseum_extract_answer(raw_text)
        return clean_answer(clean_think_tags(ans)) if ans else ""
    
    # Extract boxed content (GLM special markers or LaTeX \boxed{})
    text = (glm_extract_boxed(raw_text) if "glm" in tag.lower() else None) \
           or extract_final_boxed_content(raw_text) \
           or raw_text
    
    # Remove think tags and validate
    text = clean_think_tags(text)
    if is_only_think_fragment(text):
        return ""
    
    return clean_answer(text)

def run_inference(
    models: List[Any],
    entry: QAEntry,
    *,
    inference_mode: str = "reason",  # "qa" or "reason"
    q_idx: Optional[int] = None,
    dataset: Optional[str] = None,
) -> ReasoningResult:
    """
    Run generative reasoning for each model on a single (image, question) pair.

    Args:
        models: List of model wrappers with .tag and .answer() methods
        entry: QAEntry with image_path, question, and optional answers
        inference_mode: "qa" (direct QA) or "reason" (step-by-step reasoning)
        q_idx: Optional question index
        dataset: Evaluated dataset
        
    Returns:
        ReasoningResult with per-model reasoning and normalized answers
    """
    img_path = entry.image_path
    question = entry.question
    gt_list = entry.answers or []

    print(f"Question: {question}")
    
    per_model: Dict[str, PerModelRecord] = {}

    for model in models:
        tag = model.tag
        
        prompt_template = get_prompt(
            task=inference_mode,  # "qa" or "reason"
            dataset=dataset, 
            model_tag=tag
        )

        raw_text = model.answer(
            img_path, 
            question, 
            prompt_tpl=prompt_template,
            return_prompt=True
        )
        answer = _post_process_raw_response(raw_text, tag, dataset=dataset, inference_mode=inference_mode)

        per_model[tag] = PerModelRecord(
            reasoning=raw_text,
            answer=answer,
        )

        torch.cuda.empty_cache()

    return ReasoningResult(
        image_path=img_path,
        question=question,
        models_reasoning=per_model,
        ground_truths=gt_list,
    )

def run_prefill(
    models: List[Any],
    entry: QAEntry,
    *,
    mode: str = "decode",
    source: str = "models_reasoning",
    running_model: Optional[str] = None,
    dataset: Optional[str] = None,
) -> PrefillResult:
    """
    Compute self/cross prefill PPL scores.

    Args:
        models: List of PrefillingModel instances with .tag and .prefill_nll()
        entry: QAEntry with answers data in .extra[source] 
        mode: "decode" (generate then score) or "cross" (score existing answers)
        source: Key to read existing answers from entry.extra
        running_model: Specific model for targeted cross-evaluation
        dataset: Evaluated dataset

    Returns:
        PrefillResult with self_ppl and optional cross_ppl scores
    """
    img_path = entry.image_path
    question = entry.question
    print(f"Prefill for: {question}")

    # Use existing answers for cross-evaluation
    extra = getattr(entry, "extra", {})
    answers = extra.get(source) or extra.get("models_reasoning")
        
    if not answers:
        raise ValueError(f"No answers found in entry.extra['{source}'] for cross prefill")

    # Self PPL
    self_ppl = {}
    for model in models:
        if model.tag in answers:
            ppl = model.prefill_nll(img_path, question, answers[model.tag]["answer"])
            self_ppl[model.tag] = ppl
            print(f"[Self-PPL] {model.tag}: {ppl:.4f}")

    # Cross PPL
    cross_ppl = {}
        
    if running_model:
        # Single model cross-evaluation
        eval_models = [name for name in answers.keys() if name != running_model]
        for answer_source in eval_models:
            for model in models:
                if model.tag == running_model:
                    ppl = model.prefill_nll(img_path, question, answers[answer_source]["answer"])
                    cross_ppl.setdefault(answer_source, {})[running_model] = ppl
                    print(f"[Cross-PPL] {running_model} on {answer_source}: {ppl:.4f}")
    else:
        # Full cross-evaluation matrix
        for answer_source in answers.keys():
            for model in models:
                if model.tag != answer_source:  # Skip self-evaluation
                    ppl = model.prefill_nll(img_path, question, answers[answer_source]["answer"])
                    cross_ppl.setdefault(answer_source, {})[model.tag] = ppl
                    print(f"[Cross-PPL] {model.tag} on {answer_source}: {ppl:.4f}")

    return PrefillResult(
        image_path=img_path,
        question=question,
        models=[m.tag for m in models],
        answers=answers,
        self_ppl=self_ppl,
        cross_ppl=cross_ppl,
    )