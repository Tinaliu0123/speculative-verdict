#!/usr/bin/env python3
"""
Render Markdown files to PNG images using wkhtmltoimage.
"""
import argparse
import pathlib
import subprocess
import tempfile
from typing import Optional
from tqdm import tqdm

def render_markdown_to_png(
    markdown_dir: pathlib.Path,
    output_dir: Optional[pathlib.Path] = None,
    width: Optional[int] = None,
    quality: int = 100
):
    """Convert all .md files in directory to .png images."""
    if output_dir is None:
        output_dir = markdown_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    md_files = list(markdown_dir.glob("*.md"))
    
    for md_file in tqdm(md_files, desc="Rendering"):
        stem = md_file.stem
        html_file = md_file.parent / f"{stem}.tmp.html"
        png_file = output_dir / f"{stem}.png"
        
        try:
            # Markdown → HTML
            subprocess.run(
                ["pandoc", "-s", str(md_file), "-o", str(html_file)],
                check=True,
                capture_output=True
            )
            
            # HTML → PNG
            cmd = [
                "wkhtmltoimage",
                "--enable-local-file-access",
                "--disable-smart-width",
                "--quality", str(quality)
            ]
            if width:
                cmd.extend(["--width", str(width)])
            
            cmd.extend([f"file://{html_file.resolve()}", str(png_file)])
            
            subprocess.run(cmd, check=True, capture_output=True)
            
        except subprocess.CalledProcessError as e:
            print(f"Error rendering {md_file.name}: {e}")
        finally:
            html_file.unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Render Markdown to PNG")
    parser.add_argument("--input", required=True, help="Directory with .md files")
    parser.add_argument("--output", help="Output directory (default: same as input)")
    parser.add_argument("--width", type=int, help="Image width in pixels")
    parser.add_argument("--quality", type=int, default=100, help="PNG quality")
    args = parser.parse_args()
    
    render_markdown_to_png(
        markdown_dir=pathlib.Path(args.input),
        output_dir=pathlib.Path(args.output) if args.output else None,
        width=args.width,
        quality=args.quality
    )
    print("Rendering complete")

if __name__ == "__main__":
    main()