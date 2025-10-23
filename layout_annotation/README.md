# Layout Annotation

Converts images to layout-annotated versions using OCR for information-intensive benchmarks.

## Requirements
```bash
# Install PaddleOCR (not included in main requirements.txt)
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install paddleocr
```

## Pipeline

1. **OCR Extraction** - Extract text/layout with PaddleOCR
2. **Markdown Generation** - Structure info as Markdown  
3. **PNG Rendering** - Render Markdown as annotated image

## Usage

```bash
python layout_annotation/pipeline.py \
    --input data/dataset_test.json \
    --output data/annotations/dataset \
    --device cpu
```

**Parameters:**
- `--input`: Input JSON/JSONL file with image paths
- `--output`: Output directory for annotated images
- `--device`: cpu or cuda (default: cpu)