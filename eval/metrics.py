#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common evaluation metrics for visual question answering benchmarks.
"""

from typing import List, Optional
from rapidfuzz.distance import Levenshtein


def compute_anls(pred: str, gts: List[str], threshold: float = 0.5) -> float:
    """
    Compute Average Normalized Levenshtein Similarity (ANLS).
    
    Args:
        pred: Predicted answer string
        gts: List of ground truth answer strings
        threshold: Score threshold (default: 0.5)
    
    Returns:
        ANLS score (0.0 to 1.0)
    """
    if not gts:
        return 0.0
    
    pred_lower = pred.lower()
    best_score = 0.0
    
    for gt in gts:
        gt_lower = gt.lower()
        max_len = max(len(pred_lower), len(gt_lower), 1)
        dist = Levenshtein.distance(pred_lower, gt_lower)
        score = 1.0 - dist / max_len
        
        # Official rule: scores below threshold count as 0
        if score < threshold:
            score = 0.0
        
        best_score = max(best_score, score)
    
    return best_score


def compute_exact_match(pred: str, gts: List[str], case_sensitive: bool = False) -> float:
    """
    Compute exact match score.
    
    Args:
        pred: Predicted answer string
        gts: List of ground truth answer strings
        case_sensitive: Whether to use case-sensitive matching
    
    Returns:
        1.0 if exact match found, else 0.0
    """
    if not gts:
        return 0.0
    
    pred_proc = pred.strip() if case_sensitive else pred.strip().lower()
    
    for gt in gts:
        gt_proc = gt.strip() if case_sensitive else gt.strip().lower()
        if pred_proc == gt_proc:
            return 1.0
    
    return 0.0