#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
InfoVQA benchmark evaluator.
Uses ANLS (Average Normalized Levenshtein Similarity) and Exact Match.
"""

from typing import List, Dict, Any
from utils import load_json_or_jsonl, extract_answer, extract_ground_truths
from metrics import compute_anls, compute_exact_match


class InfoVQAEvaluator:
    """Evaluator for InfoVQA benchmark."""
    
    def __init__(
        self,
        answer_key: str = 'final_answer',
        anls_threshold: float = 0.5,
        include_em: bool = True
    ):
        """
        Initialize InfoVQA evaluator.
        
        Args:
            answer_key: Key to extract predictions from (default: 'final_answer')
            anls_threshold: ANLS threshold (default: 0.5)
            include_em: Whether to also compute exact match (default: True)
        """
        self.answer_key = answer_key
        self.anls_threshold = anls_threshold
        self.include_em = include_em
    
    def evaluate_file(self, input_path: str) -> Dict[str, Any]:
        """
        Evaluate predictions from a file.
        
        Args:
            input_path: Path to predictions file (JSON or JSONL)
        
        Returns:
            Dictionary with evaluation results
        """
        data = load_json_or_jsonl(input_path)
        
        anls_scores = []
        em_scores = []
        
        for entry in data:
            gts = extract_ground_truths(entry)
            pred = extract_answer(entry, self.answer_key)
            
            # Compute ANLS
            anls = compute_anls(pred, gts, self.anls_threshold)
            anls_scores.append(anls)
            
            # Compute EM if requested
            if self.include_em:
                em = compute_exact_match(pred, gts)
                em_scores.append(em)
        
        # Build results
        results = {
            'num_samples': len(data),
            'anls': sum(anls_scores) / len(anls_scores) if anls_scores else 0.0,
        }
        
        if self.include_em:
            results['exact_match'] = sum(em_scores) / len(em_scores) if em_scores else 0.0
        
        return results
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print evaluation results."""
        print(f"\n{'='*60}")
        print(f"InfoVQA Evaluation Results")
        print(f"{'='*60}")
        print(f"Total samples: {results['num_samples']}")
        print(f"ANLS:          {results['anls']:.4f}")
        
        if 'exact_match' in results:
            print(f"Exact Match:   {results['exact_match']:.4f}")
        
        print(f"{'='*60}\n")