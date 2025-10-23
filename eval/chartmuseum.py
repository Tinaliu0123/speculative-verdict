#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChartMuseum benchmark evaluator.
Uses LLM as a judge to compare predicted and ground truth answers.
"""

from typing import List, Dict, Any
import logging

from utils import load_json_or_jsonl, extract_answer, extract_ground_truths

# Prompt template for LLM judge
COMPARE_ANSWER_PROMPT = """You are an expert evaluator for question answering systems.

Given a question and two answers, determine if they are semantically equivalent.

Question: [QUESTION]

Answer 1 (Ground Truth): [ANSWER1]

Answer 2 (Prediction): [ANSWER2]

Consider the answers equivalent if they convey the same essential information, even if worded differently. Ignore minor differences in formatting, punctuation, or phrasing.

Respond with only "yes" if they are equivalent, or "no" if they are not."""


class ChartMuseumEvaluator:
    """
    Evaluator for ChartMuseum benchmark using LLM as judge.
    
    Requires: pip install bespokelabs-curator datasets
    """
    
    def __init__(
        self,
        answer_key: str = 'final_answer',
        model_name: str = 'gpt-4.1-mini-2025-04-14',
        working_dir: str = './cache/chartmuseum',
        temperature: float = 0.0
    ):
        """
        Initialize ChartMuseum evaluator.
        
        Args:
            answer_key: Key to extract predictions from (default: 'final_answer')
            model_name: LLM model name for judging (default: gpt-4.1-mini)
            working_dir: Directory for caching LLM responses
            temperature: LLM temperature (default: 0 for deterministic)
        """
        self.answer_key = answer_key
        self.model_name = model_name
        self.working_dir = working_dir
        self.temperature = temperature
        self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM judge."""
        try:
            from bespokelabs import curator
            
            class AnswerCompareGenerator(curator.LLM):
                def prompt(self, input: dict) -> str:
                    return COMPARE_ANSWER_PROMPT.replace(
                        '[QUESTION]', input['question']
                    ).replace(
                        '[ANSWER1]', input['answer1']
                    ).replace(
                        '[ANSWER2]', input['answer2']
                    )
                
                def parse(self, input: dict, response) -> dict:
                    input['equal_answer'] = 1 if 'yes' in response.lower() else 0
                    return input
            
            config = {
                'model_name': self.model_name,
                'generation_params': {'temperature': self.temperature},
                'backend_params': {
                    'max_requests_per_minute': 12_000,
                    'max_tokens_per_minute': 4_000_000,
                    'seconds_to_pause_on_rate_limit': 15.0,
                },
            }
            self.llm_judge = AnswerCompareGenerator(**config)
            
        except ImportError:
            raise ImportError(
                "ChartMuseum evaluator requires bespokelabs-curator. "
                "Install with: pip install bespokelabs-curator datasets"
            )
    
    def evaluate_file(self, input_path: str) -> Dict[str, Any]:
        """
        Evaluate using LLM judge.
        
        Args:
            input_path: Path to predictions file (JSON or JSONL)
        
        Returns:
            Results with accuracy score
        """
        try:
            from datasets import Dataset
            import pandas as pd
        except ImportError:
            raise ImportError(
                "ChartMuseum evaluator requires datasets. "
                "Install with: pip install datasets"
            )
        
        data = load_json_or_jsonl(input_path)
        
        # Prepare dataset
        questions = []
        gts_str = []
        preds = []
        
        for entry in data:
            questions.append(entry.get('question', ''))
            gts = extract_ground_truths(entry)
            gts_str.append(str(gts))  # Convert list to string for prompt
            pred = extract_answer(entry, self.answer_key)
            preds.append(pred)
        
        # Create dataset for LLM judge
        ds = Dataset.from_dict({
            'question': questions,
            'answer1': gts_str,
            'answer2': preds,
        })
        
        logging.info(f"Evaluating {len(ds)} samples with LLM judge ({self.model_name})...")
        response = self.llm_judge(ds, working_dir=self.working_dir)
        
        # Extract results
        equal_flags = self._extract_equal_flags(response)
        
        results = {
            'num_samples': len(data),
            'accuracy': sum(equal_flags) / len(equal_flags) if equal_flags else 0.0,
        }
        
        return results
    
    @staticmethod
    def _extract_equal_flags(response) -> List[int]:
        """Extract equal_answer flags from curator response."""
        import pandas as pd
        
        # Handle different response formats from curator
        if hasattr(response, 'to_pandas'):
            df = response.to_pandas()
        elif hasattr(response, 'to_dataframe'):
            df = response.to_dataframe()
        elif hasattr(response, 'dataset'):
            hf_ds = response.dataset
            df = hf_ds.to_pandas() if hasattr(hf_ds, 'to_pandas') else pd.DataFrame(hf_ds)
        elif isinstance(response, pd.DataFrame):
            df = response
        else:
            df = pd.DataFrame(response)
        
        return df['equal_answer'].astype(int).tolist()
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print ChartMuseum results."""
        print(f"\n{'='*60}")
        print(f"ChartMuseum Evaluation Results")
        print(f"{'='*60}")
        print(f"LLM Judge:     {self.model_name}")
        print(f"Total samples: {results['num_samples']}")
        print(f"Accuracy:      {results['accuracy']:.4f}")
        print(f"{'='*60}\n")