#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified evaluation CLI for visual question answering benchmarks.

Usage:
    # InfoVQA
    python eval.py infovqa --input preds.json
    
    # ChartQAPro (single-file mode)
    python eval.py chartqapro --input preds.json
    
    # ChartQAPro (two-file mode with metadata)
    python eval.py chartqapro --input preds.json --meta meta.jsonl
    
    # ChartMuseum
    python eval.py chartmuseum --input preds.json

    # HR-Bench
    python eval.py hrbench --input preds.json --bench hr_bench_4k.jsonl
"""

import argparse
import logging
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from infovqa import InfoVQAEvaluator
from chartqapro import ChartQAProEvaluator
from chartmuseum import ChartMuseumEvaluator
from hrbench import HRBenchEvaluator

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )

def save_results(results: dict, output_path: str):
    """Save results to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def evaluate_infovqa(args):
    """Run InfoVQA evaluation."""
    evaluator = InfoVQAEvaluator(
        answer_key=args.answer_key,
        anls_threshold=args.anls_threshold,
        include_em=args.include_em
    )
    
    results = evaluator.evaluate_file(args.input)
    evaluator.print_results(results)
    
    if args.output:
        save_results(results, args.output)
        logging.info(f"Results saved to {args.output}")
    
    return results

def evaluate_chartqapro(args):
    """Run ChartQAPro evaluation."""
    evaluator = ChartQAProEvaluator(
        answer_key=args.answer_key,
        max_relative_change=args.tolerance,
        anls_threshold=args.anls_threshold
    )
    
    results = evaluator.evaluate_file(
        input_path=args.input,
        meta_path=args.meta
    )
    
    evaluator.print_results(results)
    
    if args.output:
        save_results(results, args.output)
        logging.info(f"Results saved to {args.output}")
    
    return results

def evaluate_chartmuseum(args):
    """Run ChartMuseum evaluation."""
    evaluator = ChartMuseumEvaluator(
        answer_key=args.answer_key,
        model_name=args.model_name,
        working_dir=args.cache_dir,
        temperature=args.temperature
    )
    
    results = evaluator.evaluate_file(args.input)
    evaluator.print_results(results)
    
    if args.output:
        save_results(results, args.output)
        logging.info(f"Results saved to {args.output}")
    
    return results

def evaluate_hrbench(args):
    """Run HR-Bench evaluation."""
    evaluator = HRBenchEvaluator(
        bench_path=args.bench,
        answer_key=args.answer_key,
        match_key=args.match_key
    )
    
    results = evaluator.evaluate_file(args.input)
    evaluator.print_results(results)
    
    if args.output:
        save_results(results, args.output)
        logging.info(f"Results saved to {args.output}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate VQA benchmark predictions')
    
    # Global arguments
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    # Subcommands for each benchmark
    subparsers = parser.add_subparsers(dest='benchmark', required=True,
                                       help='Benchmark to evaluate')
    
    # InfoVQA
    infovqa_parser = subparsers.add_parser('infovqa', help='Evaluate InfoVQA')
    infovqa_parser.add_argument(
        '--input', '-i', required=True,
        help='Path to predictions file (JSON/JSONL)'
    )
    infovqa_parser.add_argument(
        '--answer-key', default='final_answer',
        help='Key for prediction answers (default: final_answer)'
    )
    infovqa_parser.add_argument(
        '--anls-threshold', type=float, default=0.5,
        help='ANLS threshold (default: 0.5)'
    )
    infovqa_parser.add_argument(
        '--no-em', dest='include_em', action='store_false',
        help='Disable exact match computation'
    )
    infovqa_parser.add_argument(
        '--output', '-o',
        help='Path to save results (JSON)'
    )
    
    # ChartQAPro
    chartqapro_parser = subparsers.add_parser('chartqapro', help='Evaluate ChartQAPro')
    chartqapro_parser.add_argument(
        '--input', '-i', required=True,
        help='Path to predictions file'
    )
    chartqapro_parser.add_argument(
        '--meta', '-m',
        help='Path to metadata file (for two-file mode, optional)'
    )
    chartqapro_parser.add_argument(
        '--answer-key', default='final_answer',
        help='Key for prediction answers (default: final_answer)'
    )
    chartqapro_parser.add_argument(
        '--tolerance', type=float, default=0.05,
        help='Numeric tolerance (default: 0.05 = 5%%)'
    )
    chartqapro_parser.add_argument(
        '--anls-threshold', type=float, default=0.5,
        help='ANLS threshold for text (default: 0.5)'
    )
    chartqapro_parser.add_argument(
        '--output', '-o',
        help='Path to save results (JSON)'
    )
    
    # ChartMuseum
    chartmuseum_parser = subparsers.add_parser('chartmuseum', help='Evaluate ChartMuseum')
    chartmuseum_parser.add_argument(
        '--input', '-i', required=True,
        help='Path to predictions file'
    )
    chartmuseum_parser.add_argument(
        '--answer-key', default='final_answer',
        help='Key for prediction answers (default: final_answer)'
    )
    chartmuseum_parser.add_argument(
        '--model-name', default='gpt-4.1-mini-2025-04-14',
        help='LLM judge model name (default: gpt-4.1-mini)'
    )
    chartmuseum_parser.add_argument(
        '--cache-dir', default='./cache/chartmuseum',
        help='Cache directory for LLM responses'
    )
    chartmuseum_parser.add_argument(
        '--temperature', type=float, default=0.0,
        help='LLM temperature (default: 0.0)'
    )
    chartmuseum_parser.add_argument(
        '--output', '-o',
        help='Path to save results (JSON)'
    )

    # HR-Bench
    hrbench_parser = subparsers.add_parser('hrbench', help='Evaluate HR-Bench')
    hrbench_parser.add_argument(
        '--input', '-i', required=True,
        help='Path to predictions file (JSON/JSONL)'
    )
    hrbench_parser.add_argument(
        '--bench', '-b',
        help='Path to benchmark JSONL file (optional if GT in predictions)'
    )
    hrbench_parser.add_argument(
        '--answer-key', default='final_answer',
        help='Key for prediction answers (default: final_answer)'
    )
    hrbench_parser.add_argument(
        '--match-key', default='idx',
        help='Key for matching records (default: idx)'
    )
    hrbench_parser.add_argument(
        '--output', '-o',
        help='Path to save results (JSON)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Run appropriate evaluator
    if args.benchmark == 'infovqa':
        return evaluate_infovqa(args)
    elif args.benchmark == 'chartqapro':
        return evaluate_chartqapro(args)
    elif args.benchmark == 'chartmuseum':
        return evaluate_chartmuseum(args)
    elif args.benchmark == 'hrbench':
        return evaluate_hrbench(args)


if __name__ == '__main__':
    main()