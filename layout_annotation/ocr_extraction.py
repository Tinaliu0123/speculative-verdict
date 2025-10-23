#!/usr/bin/env python3
"""
Extract text and layout structure from images using PaddleOCR PPStructure.
Outputs JSON and Markdown files.
"""
import argparse
import json
import pathlib
from typing import List, Optional
from tqdm import tqdm
from paddleocr import PPStructureV3

def load_image_paths(
    json_file: Optional[str] = None,
    glob_pattern: Optional[str] = None,
    json_field: str = "samples",
    image_key: str = "image_path",
    limit: Optional[int] = None
) -> List[str]:
    """Load image paths from JSON or glob pattern."""
    if json_file:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        entries = data if isinstance(data, list) else data.get(json_field, [])
        if limit:
            entries = entries[:limit]
        
        paths = [entry[image_key] for entry in entries]
        return list(dict.fromkeys(paths))  # Deduplicate
    
    elif glob_pattern:
        import glob
        paths = sorted(glob.glob(glob_pattern))
        if limit:
            paths = paths[:limit]
        return list(dict.fromkeys(paths))
    
    else:
        raise ValueError("Must provide either json_file or glob_pattern")

def process_images(
    image_paths: List[str],
    output_dir: pathlib.Path,
    device: str = "cpu",
    skip_existing: bool = True
):
    """Run PPStructure OCR on images and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ocr = PPStructureV3(
        device=device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_recognition_model_name="en_PP-OCRv4_mobile_rec"
    )
    
    for img_path in tqdm(image_paths, desc="OCR Extraction"):
        stem = pathlib.Path(img_path).stem
        json_out = output_dir / f"{stem}.json"
        
        if skip_existing and json_out.exists():
            continue
        
        try:
            for page in ocr.predict(img_path):
                page.save_to_json(output_dir)
                page.save_to_markdown(output_dir)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Batch OCR extraction with PPStructure")
    parser.add_argument("--json", help="Input JSON file with image paths")
    parser.add_argument("--glob", help="Glob pattern for images (e.g., 'imgs/*.png')")
    parser.add_argument("--field", default="samples", help="JSON field name")
    parser.add_argument("--key", default="input_image", help="Key for image path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--limit", type=int, help="Limit number of images")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    args = parser.parse_args()
    
    image_paths = load_image_paths(
        json_file=args.json,
        glob_pattern=args.glob,
        json_field=args.field,
        image_key=args.key,
        limit=args.limit
    )
    
    print(f"Processing {len(image_paths)} images...")
    process_images(
        image_paths=image_paths,
        output_dir=pathlib.Path(args.output),
        device=args.device
    )
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()