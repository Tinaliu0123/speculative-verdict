#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChartQAPro benchmark evaluator.
Uses relaxed correctness with numeric tolerance and ANLS for text.
"""

import re
import ast
from typing import List, Dict, Any, Optional
from collections import defaultdict

from utils import load_json_or_jsonl, extract_answer, extract_ground_truths
from metrics import compute_anls


class ChartQAProEvaluator:
    """Evaluator for ChartQAPro benchmark."""
    
    def __init__(
        self,
        answer_key: str = 'final_answer',
        max_relative_change: float = 0.05,
        anls_threshold: float = 0.5
    ):
        """
        Initialize ChartQAPro evaluator.
        
        Args:
            answer_key: Key to extract predictions from (default: 'final_answer')
            max_relative_change: Tolerance for numeric comparisons (default: 5%)
            anls_threshold: ANLS threshold for text comparisons (default: 0.5)
        """
        self.answer_key = answer_key
        self.max_relative_change = max_relative_change
        self.anls_threshold = anls_threshold
    
    def evaluate_file(
        self,
        input_path: str,
        meta_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate ChartQAPro predictions.
        
        Args:
            input_path: Path to predictions file
            meta_path: Optional path to metadata file (for two-file mode)
        
        Returns:
            Results with overall and per-split scores
        """
        data = load_json_or_jsonl(input_path)
        
        # Load metadata if provided (two-file mode)
        if meta_path:
            meta_data = load_json_or_jsonl(meta_path)
            assert len(data) == len(meta_data), \
                f"Length mismatch: preds={len(data)}, meta={len(meta_data)}"
        else:
            # Single-file mode: data contains both predictions and metadata
            meta_data = data
        
        # Accumulate scores by question type
        scores_by_split = defaultdict(list)
        
        # Process each entry
        for entry, meta in zip(data, meta_data):
            gts = extract_ground_truths(meta)
            pred = extract_answer(entry, self.answer_key)
            
            question_type = meta.get('Question Type', 'Unknown')
            year_flag = meta.get('Year')
            
            # Special handling for Conversational questions
            if question_type == 'Conversational' and isinstance(year_flag, list):
                year_flag = year_flag[-1:]
            
            # Compute score
            score = self._compute_score(pred, gts, question_type, year_flag)
            scores_by_split[question_type].append(score)
        
        # Calculate averages
        results = {
            'num_samples': len(data),
            'by_split': {
                split: sum(scores) / len(scores) if scores else 0.0
                for split, scores in scores_by_split.items()
            }
        }
        
        # Add overall score
        all_scores = [s for scores in scores_by_split.values() for s in scores]
        results['overall'] = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        return results
    
    def _compute_score(
        self,
        pred: str,
        gts: List[str],
        question_type: str,
        year_flag: Any
    ) -> float:
        """Compute relaxed correctness score."""
        if not gts:
            return 0.0
        
        # Use last ground truth (ChartQAPro convention)
        target = str(gts[-1]).strip()
        
        # Fact Checking and Multi Choice always use exact match
        use_exact = (question_type in ['Fact Checking', 'Multi Choice'])
        
        return self._relaxed_correctness(target, pred, year_flag, use_exact)
    
    def _relaxed_correctness(
        self,
        target: str,
        prediction: str,
        year_flag: Any = None,
        always_use_exact: bool = False
    ) -> float:
        """Compute relaxed correctness with support for lists and numeric values."""
        # Parse potential list formats
        target_list = self._parse_to_list(target) or [target]
        pred_list = self._parse_to_list(prediction) or [prediction]
        n = max(len(target_list), len(pred_list))
        
        # Normalize year flags to boolean list
        year_flags = self._normalize_year_flags(year_flag, n)
        
        # Compute score for each element
        scores = []
        for i in range(n):
            if i >= len(target_list) or i >= len(pred_list):
                scores.append(0.0)
                continue
            
            t_item = str(target_list[i]).strip()
            p_item = str(pred_list[i]).strip()
            
            if year_flags[i] or always_use_exact:
                # Exact match (case-insensitive)
                scores.append(1.0 if t_item.lower() == p_item.lower() else 0.0)
            else:
                # Numeric or ANLS
                scores.append(self._evaluate_single(t_item, p_item))
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _evaluate_single(self, target: str, prediction: str) -> float:
        """Evaluate single target-prediction pair."""
        t = target.strip('%').strip()
        p = prediction.strip('%').strip()
        
        # Try numeric comparison
        t_num = self._to_float(t)
        p_num = self._to_float(p)
        
        if t_num is not None and p_num is not None:
            if t_num == 0.0:
                return 1.0 if p_num == 0.0 else 0.0
            change = abs(p_num - t_num) / abs(t_num)
            return 1.0 if change <= self.max_relative_change else 0.0
        
        # Fallback to ANLS
        return compute_anls(p.lower(), [t.lower()], self.anls_threshold)
    
    @staticmethod
    def _to_float(text: str) -> Optional[float]:
        """Convert text to float."""
        try:
            return float(text)
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _parse_to_list(text: str) -> Optional[List[str]]:
        """Parse string representation of list."""
        if not isinstance(text, str):
            return None
        
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x).strip(" '\"") for x in parsed]
        except:
            pass
        
        # Try fixing format [a, b] -> ['a', 'b']
        match = re.match(r"^\[(.*)\]$", text.strip())
        if match:
            content = match.group(1)
            corrected = re.sub(r"(?<!['\w])(\w[^,]*?)(?!['\w])", r"'\1'", content)
            try:
                parsed = ast.literal_eval(f"[{corrected}]")
                if isinstance(parsed, list):
                    return [str(x).strip(" '\"") for x in parsed]
            except:
                pass
        
        return None
    
    @staticmethod
    def _normalize_year_flags(year_flag: Any, length: int) -> List[bool]:
        """Normalize year flag to boolean list of specified length."""
        if year_flag is None:
            return [False] * length
        
        # Convert to list
        if not isinstance(year_flag, list):
            year_list = [year_flag]
        else:
            year_list = year_flag
        
        # Extend to length
        if len(year_list) < length:
            year_list = (year_list * length)[:length]
        
        # Convert to bool
        def to_bool(x):
            if isinstance(x, str):
                return x.strip().upper() == 'YES'
            return bool(x)
        
        return [to_bool(x) for x in year_list]
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print ChartQAPro results by split."""
        print(f"\n{'='*60}")
        print(f"ChartQAPro Evaluation Results")
        print(f"{'='*60}")
        print(f"Total samples: {results['num_samples']}")
        print(f"Overall:       {results['overall']:.4f}")
        
        print(f"\nBy Question Type:")
        for split, score in sorted(results['by_split'].items()):
            print(f"  {split:20s}: {score:.4f}")
        
        print(f"{'='*60}\n")