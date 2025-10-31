#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HR-Bench evaluator for high-resolution document VQA.

Supports:
- Binary accuracy for multiple-choice questions (A/B/C/D)
- Category-level and per-model statistics
"""

import re
import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict
from pathlib import Path

from metrics import compute_exact_match
from utils import load_json_or_jsonl, extract_answer


logger = logging.getLogger(__name__)


def extract_option(text: str) -> str:
    """Extract multiple-choice option (A/B/C/D) from text."""
    text = str(text).strip().upper()
    
    # Direct match
    if text in ['A', 'B', 'C', 'D']:
        return text
    
    # Extract from text
    match = re.search(r'\b([ABCD])\b', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    return text


class HRBenchEvaluator:
    """Evaluator for HR-Bench 4K/8K benchmarks."""
    
    def __init__(
        self,
        bench_path: Optional[str] = None,
        answer_key: str = 'final_answer',
        match_key: str = 'idx'
    ):
        """
        Initialize HR-Bench evaluator.
        
        Args:
            bench_path: Path to benchmark JSONL file (optional if GT in preds)
            answer_key: Key for prediction answers
            match_key: Key for matching records (idx, id, question)
        """
        self.answer_key = answer_key
        self.match_key = match_key
        self.benchmark = {}
        
        if bench_path:
            self._load_benchmark(bench_path)
    
    def _load_benchmark(self, path: str):
        """Load benchmark file with ground truths."""
        logger.info(f"Loading benchmark from {path}")
        
        data = load_json_or_jsonl(path)
        for entry in data:
            # This will be used to match with prediction entries
            key = self._normalize_key(entry.get(self.match_key))
            self.benchmark[key] = {
                'category': entry.get('category', 'unknown'),
                'answers': [a.upper() for a in entry.get('answers', [])]
            }
        
        logger.info(f"Loaded {len(self.benchmark)} benchmark entries")
    
    def _normalize_key(self, key: Any) -> str:
        """Normalize matching key."""
        if key is None:
            return ''
        return str(key).strip()
    
    def _compute_accuracy(self, pred: str, gts: List[str]) -> float:
        """Compute binary accuracy for multiple choice."""
        pred_option = extract_option(pred)
        return 1.0 if pred_option in gts else 0.0
    
    def evaluate_file(self, input_path: str) -> Dict[str, Any]:
        """
        Evaluate predictions file.
        
        Args:
            input_path: Path to predictions (JSON/JSONL)
        
        Returns:
            Results dictionary with overall and per-category metrics
        """
        logger.info(f"Evaluating {input_path}")
        
        preds = load_json_or_jsonl(input_path)
        
        # Aggregators
        cat_scores = defaultdict(list)
        cat_model_scores = defaultdict(lambda: defaultdict(list))
        total_scores = []
        model_scores = defaultdict(list)
        
        for pred_entry in preds:
            key = self._normalize_key(pred_entry.get(self.match_key))
            
            # Get ground truths
            if self.benchmark:
                if key not in self.benchmark:
                    logger.warning(f"Key not found in benchmark: {key}")
                    continue
                info = self.benchmark[key]
            else:
                # GT in predictions
                info = {
                    'category': pred_entry.get('category', 'unknown'),
                    'answers': [a.upper() for a in pred_entry.get('answers', [])]
                }

            # print(info)
            
            category = info['category']
            gts = info['answers']
            
            # Overall prediction
            pred_ans = extract_answer(pred_entry, self.answer_key)
            acc = self._compute_accuracy(pred_ans, gts)
            
            cat_scores[category].append(acc)
            total_scores.append(acc)
            
            # Per-model predictions
            if 'models_reasoning' in pred_entry:
                for model_name, model_data in pred_entry['models_reasoning'].items():
                    if isinstance(model_data, dict):
                        model_ans = model_data.get('answer', '')
                    else:
                        model_ans = str(model_data)
                    
                    model_acc = self._compute_accuracy(model_ans, gts)
                    cat_model_scores[category][model_name].append(model_acc)
                    model_scores[model_name].append(model_acc)
        
        # Compute results
        results = {
            'overall': {
                'accuracy': self._mean(total_scores),
                'count': len(total_scores)
            },
            'by_category': {
                cat: {
                    'accuracy': self._mean(scores),
                    'count': len(scores)
                }
                for cat, scores in cat_scores.items()
            }
        }
        
        # Per-model results
        if model_scores:
            results['by_model'] = {
                model: {
                    'accuracy': self._mean(scores),
                    'count': len(scores)
                }
                for model, scores in model_scores.items()
            }
            
            # Category × Model breakdown
            results['by_category_model'] = {}
            for cat, models in cat_model_scores.items():
                results['by_category_model'][cat] = {
                    model: {
                        'accuracy': self._mean(scores),
                        'count': len(scores)
                    }
                    for model, scores in models.items()
                }
        
        return results
    
    def _mean(self, values: List[float]) -> float:
        """Compute mean."""
        return sum(values) / len(values) if values else 0.0
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted results."""
        print("\n" + "=" * 70)
        print("HR-Bench Evaluation Results")
        print("=" * 70)
        
        # Overall
        overall = results['overall']
        print(f"\nOverall Accuracy: {overall['accuracy']:.4f} (n={overall['count']})")
        
        # By category
        print("\n" + "-" * 70)
        print("By Category:")
        print("-" * 70)
        for cat in sorted(results['by_category'].keys()):
            cat_data = results['by_category'][cat]
            print(f"{cat:30s}: {cat_data['accuracy']:.4f} (n={cat_data['count']})")
        
        # By model
        if 'by_model' in results:
            print("\n" + "-" * 70)
            print("By Model:")
            print("-" * 70)
            for model in sorted(results['by_model'].keys()):
                model_data = results['by_model'][model]
                print(f"{model:30s}: {model_data['accuracy']:.4f} (n={model_data['count']})")
        
        # Category × Model
        if 'by_category_model' in results:
            print("\n" + "-" * 70)
            print("By Category and Model:")
            print("-" * 70)
            for cat in sorted(results['by_category_model'].keys()):
                print(f"\n[{cat}]")
                for model, model_data in sorted(results['by_category_model'][cat].items()):
                    print(f"  {model:28s}: {model_data['accuracy']:.4f} (n={model_data['count']})")
        
        print("=" * 70 + "\n")