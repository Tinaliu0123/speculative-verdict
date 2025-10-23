#!/usr/bin/env python3
"""
End-to-end pipeline: Image → OCR → Markdown → Annotated PNG
"""
import argparse
import pathlib
from ocr_extraction import load_image_paths, process_images
from render_markdown import render_markdown_to_png

def run_pipeline(
    input_json: str,
    output_dir: str,
    image_key: str = "image_path",
    device: str = "cpu",
    limit: int = None
):
    """Run complete annotation pipeline."""
    output_path = pathlib.Path(output_dir)
    ocr_dir = output_path / "ocr_results"
    annotated_dir = output_path / "annotated_images"
    
    print("Step 1: OCR Extraction")
    image_paths = load_image_paths(
        json_file=input_json,
        image_key=image_key,
        limit=limit
    )
    process_images(image_paths, ocr_dir, device=device)
    
    print("\nStep 2: Rendering Annotations")
    render_markdown_to_png(
        markdown_dir=ocr_dir,
        output_dir=annotated_dir
    )
    
    print(f"\nPipeline complete!")
    print(f"OCR results: {ocr_dir}")
    print(f"Annotated images: {annotated_dir}")

def main():
    parser = argparse.ArgumentParser(description="Full annotation pipeline")
    parser.add_argument("--input", required=True, help="Input JSON with image paths")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--key", default="image_path", help="JSON key for image path")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--limit", type=int, help="Limit number of images")
    args = parser.parse_args()
    
    run_pipeline(
        input_json=args.input,
        output_dir=args.output,
        image_key=args.key,
        device=args.device,
        limit=args.limit
    )

if __name__ == "__main__":
    main()