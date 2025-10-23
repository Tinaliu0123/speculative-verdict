from __future__ import annotations
from typing import List, Optional, Union
import re

# GLM-style boxed markers
_BOX_GLM_RE = re.compile(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", flags=re.DOTALL)

# LaTeX \boxed{...}
_BOX_LATEX_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

# <think> ... </think> blocks
THINK_BLOCK_RE = re.compile(r"<\s*think\s*>.*?<\s*/\s*think\s*>", flags=re.IGNORECASE | re.DOTALL)
THINK_OPEN_RE  = re.compile(r"<\s*think\s*>", flags=re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"<\s*/\s*think\s*>", flags=re.IGNORECASE)

# \text{...} and escaped percents like "\%" or "\\%"
_TEXT_CMD_RE = re.compile(r"\\text\{(.*?)\}")
_ESCAPED_PCT_RE = re.compile(r"\\\\?%")

_WHITESPACE_ONLY_RE = re.compile(r"^\s*$")


def extract_final_boxed_content(text: str) -> Optional[str]:
    """
    Extract the content of the LAST LaTeX \\boxed{...} occurrence.
    Returns None if no \\boxed is found.
    """
    if not isinstance(text, str) or not text:
        return None
    matches = _BOX_LATEX_RE.findall(text)
    return matches[-1] if matches else None

def glm_extract_boxed(text: str, return_all: bool = False) -> Optional[Union[str, List[str]]]:
    """
    Extract content from GLM box markers: <|begin_of_box|> ... <|end_of_box|>.
    If return_all=True, return a list of all matches (trimmed).
    Otherwise return the first match or None.
    """
    if not isinstance(text, str) or not text:
        return None
    if return_all:
        return [m.strip() for m in _BOX_GLM_RE.findall(text)]
    m = _BOX_GLM_RE.search(text)
    return m.group(1).strip() if m else None

def clean_think_tags(text: str) -> str:
    """
    Remove all <think>...</think> blocks, return the remaining text.
    """
    if not isinstance(text, str) or not text:
        return ""
    cleaned = THINK_BLOCK_RE.sub("", text)
    return cleaned.strip()

def is_only_think_fragment(text: str) -> bool:
    """
    Return True iff the text contains think tags but (after removing paired blocks and any
    stray open/close tags) no substantive content remains.

    This catches cases where the model outputs only hidden chain-of-thought or truncated tags.
    """
    if not isinstance(text, str):
        return False
    t = text.strip()
    if not t:
        return False

    open_cnt = len(THINK_OPEN_RE.findall(t))
    close_cnt = len(THINK_CLOSE_RE.findall(t))

    t2 = THINK_BLOCK_RE.sub("", t).strip()
    t2 = THINK_OPEN_RE.sub("", t2).strip()
    t2 = THINK_CLOSE_RE.sub("", t2).strip()

    if _WHITESPACE_ONLY_RE.match(t2):
        return (open_cnt + close_cnt) > 0 or (open_cnt != close_cnt)

    return False

def clean_answer(ans: str) -> str:
    """
    Final normalization of the extracted answer
    """
    if ans is None:
        ans = ""
    elif not isinstance(ans, str):
        ans = str(ans)

    ans = _TEXT_CMD_RE.sub(r"\1", ans)  
    ans = _ESCAPED_PCT_RE.sub("%", ans) 
    ans = ans.replace("\\", "")          
    ans = " ".join(ans.split())       
    return ans

def chartmuseum_extract_answer(text):
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, text + '</answer>', re.DOTALL)
    return answer_match.group(1).strip() if answer_match else ""