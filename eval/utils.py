#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for evaluation.
"""

import json
from typing import List, Dict, Any, Union, Optional
from pathlib import Path


def load_json_or_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load JSON array or JSONL file into a list of dictionaries.
    
    Args:
        file_path: Path to JSON or JSONL file
    
    Returns:
        List of dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # Check if it's a JSON array or JSONL
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            # JSON array
            return json.load(f)
        else:
            # JSONL
            return [json.loads(line.strip()) for line in f if line.strip()]


def extract_answer(
    entry: Dict[str, Any],
    answer_key: str = 'final_answer',
    model_key: Optional[str] = None
) -> str:
    """
    Extract answer from entry based on configuration.
    
    Args:
        entry: Data entry dictionary
        answer_key: Key to extract answer from (e.g., 'final_answer', 'answer')
        model_key: If provided, extract from models_reasoning[model_key]
    
    Returns:
        Answer string
    """
    if model_key:
        models_reasoning = entry.get('models_reasoning', {})
        if model_key in models_reasoning:
            model_data = models_reasoning[model_key]
            if isinstance(model_data, dict):
                return str(model_data.get(answer_key, '')).strip()
        return ''
    
    return str(entry.get(answer_key, '')).strip()


def extract_ground_truths(entry: Dict[str, Any]) -> List[str]:
    """
    Extract ground truth answers from entry.
    
    Args:
        entry: Data entry dictionary
    
    Returns:
        List of ground truth strings
    """
    # Try different possible keys
    for key in ['ground_truths', 'answers', 'Answer']:
        if key in entry:
            gts = entry[key]
            if isinstance(gts, list):
                return [str(gt).strip() for gt in gts]
            elif isinstance(gts, str):
                return [gts.strip()]
    
    return []


def save_results(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    format: str = 'json'
) -> None:
    """
    Save evaluation results to file.
    
    Args:
        results: Results dictionary
        output_path: Output file path
        format: Output format ('json' or 'jsonl')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        if format == 'jsonl' and isinstance(results.get('data'), list):
            for item in results['data']:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            json.dump(results, f, ensure_ascii=False, indent=2)