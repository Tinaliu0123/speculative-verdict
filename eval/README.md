# Evaluation

Evaluation toolkit for InfoVQA, ChartQAPro, and ChartMuseum benchmarks.

## Quick Start

```bash
# InfoVQA (ANLS metric)
python eval/eval.py infovqa --input preds.json

# ChartQAPro (relaxed accuracy)
python eval/eval.py chartqapro --input preds.json --meta meta.jsonl

# ChartMuseum (GPT-4 judge)
python eval/eval.py chartmuseum --input preds.json
```

## Input Format

JSON/JSONL file with:
```json
{
  "question": "What is shown?",
  "ground_truths": ["answer1", "answer2"],
  "final_answer": "predicted answer"
}
```

Use `--answer-key` to specify a different prediction field.

## Metrics

| Benchmark | Metric | Description |
|-----------|--------|-------------|
| **InfoVQA** | ANLS | Edit distance-based (threshold=0.5) |
| **ChartQAPro** | Relaxed Accuracy | Numeric: ≤5% error; Text: ANLS |
| **ChartMuseum** | GPT-4 Judge | Semantic equivalence |