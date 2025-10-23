"""
Prompt templates.

Organized into three template families:
    - QA: Short-answer questions
    - REASON: Step-by-step reasoning prompts
    - VERDICT: Final answer selection prompts

Supported datasets:
    infographicvqa, hrbench, chartmuseum, chartqapro
"""

from __future__ import annotations
from typing import Optional

# QA Prompts

# infovqa
_QA_INFOVQA = (
     "{}\nAnswer the question using a single word or phrase." 
)
# hrbench
_QA_HRBENCH = (
    "{}\nAnswer with the option letter only."
)
# museum:
_QA_MUSEUM = (
    """Please answer the question using the chart image.

Question: {}

Please first generate your reasoning process and then provide the user with the answer. Use the following format:

<think> 
... your thinking process here ... 
</think> 
<answer> 
... your final answer (entity(s) or number) ...
</answer>"""
)
# pro
_QA_PRO = (
    "{}"
)

QA_PROMPTS = {
    "infovqa": _QA_INFOVQA,
    "hrbench": _QA_HRBENCH,
    "museum":  _QA_MUSEUM,
    "pro":     _QA_PRO,
}


# Reasoning prompt

# _REASON_INFOVQA = (
#     "Question: {} Let's think step by step:"
# )
_REASON_INFOVQA = (
    "Question: {} Please think step-by-step about the image to answer the question using a single word or phrase enclosed within \\boxed{{}}."
)
_REASON_HRBENCH = _REASON_INFOVQA  
_REASON_MUSEUM  = _QA_MUSEUM
_REASON_PRO     = (
   """{}

Please first generate your reasoning process and then provide the user with the answer. Use the following format:

<think> 
... your thinking process here ... 
</think> 
<answer> 
... your final answer (entity(s) or number) ...
</answer>"""
)

REASON_PROMPTS = {
    "infovqa": _REASON_INFOVQA,
    "hrbench": _REASON_HRBENCH,
    "museum":  _REASON_MUSEUM,
    "pro":     _REASON_PRO,
}


# Verdict prompts

_VERDICT_INFO = (
    "please give the final answer using a single word or phrase enclosed within \\boxed{{}}."
)
_VERDICT_MUSEUM = _VERDICT_INFO 
_VERDICT_HR = (
    "please directly give the final answer with the option's letter enclosed within \\boxed{{}}."
)
_VERDICT_PRO = (
    "please directly give the final answer enclosed within \\boxed{{}}."
)

VERDICT_PROMPTS = {
    "infovqa": _VERDICT_INFO,
    "museum": _VERDICT_MUSEUM,
    "hrbench": _VERDICT_HR,
    "pro":        _VERDICT_PRO,
}


def get_prompt(task: str,
               dataset: str,
               *,
               model_tag: Optional[str] = None,) -> str:
    """
    Retrieve a prompt template by task + dataset, with optional model-specific override.

    Args:
        task:    {"qa", "reason", "verdict"}
        dataset: {"infovqa", "hrbench", "museum", "pro"} for QA/REASON/VERDICT;
        model_tag: optional lowercase model tag keywords (e.g., "mimo", "qwen")

    Returns:
        A format string expecting one '{}' slot for the question.
    """
    task = task.lower().strip()
    dataset = dataset.lower().strip()
    tag = (model_tag or "").lower()

    if task == "qa":
        return QA_PROMPTS[dataset]

    elif task == "reason":
        if dataset not in REASON_PROMPTS:
            raise ValueError(f"[prompts] Unknown REASON dataset: {dataset}")
        return REASON_PROMPTS[dataset]

    elif task == "verdict":
        if dataset not in VERDICT_PROMPTS:
            raise ValueError(f"[prompts] Unknown VERDICT dataset: {dataset}")
        return VERDICT_PROMPTS[dataset]

    else:
        raise ValueError(f"[prompts] Unknown task: {task}")

def detect_dataset_from_path(path: str) -> str:
    """
    Auto-detect dataset from file path or name.
    """
    path_lower = path.lower()
    
    if "hrbench" in path_lower or "hr_bench" in path_lower:
        return "hrbench"
    elif "infovqa" in path_lower or "info_vqa" in path_lower:
        return "infovqa"  
    elif "museum" in path_lower or "chart_museum" in path_lower:
        return "museum"
    elif "pro" in path_lower or "chartqa_pro" in path_lower:
        return "pro"
    else:
        return "infovqa"

def get_legacy_prompts(dataset: str) -> dict:
    """
    Get prompts in legacy format for backward compatibility.
    """
    return {
        "QA_PROMPT": get_prompt("qa", dataset),
        "REASON_PROMPT": get_prompt("reason", dataset),
    }