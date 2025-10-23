"""
Verdict stage for final answer.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
from PIL import Image
import os
import base64
import re
import logging

import torch
from qwen_vl_utils import process_vision_info

from utils.post_process import extract_final_boxed_content, clean_think_tags, clean_answer
from prompts import get_prompt

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    
def gpt4o_verdict(
    question: str,
    answers_dict: Dict[str, Dict[str, Any]],   
    orig_img_path: str,
    *,
    annotated_folder: str = "",
    crop_paths: Optional[List[str]] = None,
    dataset: str = "infovqa",
    api_key: Optional[str] = None,
    model_name: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 800,
) -> Tuple[str, str]:
    """
    Use GPT-4o model to fuse reasoning from multiple models.
    Uses dataset-specific prompts from prompts.py.
    
    Args:
        question: Question text
        answers_dict: Dict of {model_tag: {"reasoning": ..., "answer": ...}}
        orig_img_path: Path to original image
        annotated_folder: Optional folder containing layout-annotated images
        dataset: Dataset name for verdict prompt selection
        api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
        model_name: OpenAI model name to use
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Tuple of (full_response, final_answer)
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI package not available. Install with: pip install openai")

    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Provide it via 'api_key' parameter "
            "or set OPENAI_API_KEY environment variable."
        )
    
    try:
        client = OpenAI(api_key=api_key)
        
        def encode_image_to_base64(image_path: str) -> Optional[str]:
            """Encode image file to base64 string."""
            try:
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
                return base64.b64encode(image_bytes).decode("utf-8")
            except Exception as e:
                logging.error(f"Error encoding image {image_path}: {e}")
                return None
        
        def create_image_block(path: str, detail: str = "high") -> Dict[str, Any]:
            """Create image block for OpenAI API."""
            b64 = encode_image_to_base64(path)
            if b64 is None:
                raise ValueError(f"Failed to encode image: {path}")
                
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": detail,
                },
            }
        
        content = []

        # Add original image
        content.append(create_image_block(orig_img_path, detail="high"))
        
        # Add annotated image if available
        if annotated_folder:
            anno_path = Path(annotated_folder) / f"{Path(orig_img_path).stem}.png"
            if anno_path.exists():
                content.append(create_image_block(str(anno_path), detail="high"))
                logging.info(f"Added annotated image: {anno_path}")
            else:
                logging.warning(f"Annotated image not found: {anno_path}")
        
        # Add question
        content.append({
            "type": "text", 
            "text": f"Question:\n{question}\n"
        })
        
        # Add model reasoning blocks
        model_items = list(answers_dict.items())
        for i, (tag, data) in enumerate(model_items, start=1):
            content.append({
                "type": "text", 
                "text": f"--- Model {i} ({tag}) ---\n"
            })
            reasoning = data.get("reasoning", "")
            answer = data.get("answer", "")
            content.append({
                "type": "text",
                "text": f"Reasoning:\n{reasoning}\nProposed Answer: {answer}\n"
            })
        
        # Get dataset-specific verdict prompt and add instruction
        verdict_ending = get_prompt(task="verdict", dataset=dataset)
        
        if annotated_folder and Path(annotated_folder).exists():
            instruction = (
                "Given the raw image, the layout-annotated image, the question, and the reasoning from multiple models, "
                + verdict_ending
            )
        else:
            instruction = (
                "Given the image, the question, and the reasoning from multiple models, "
                + verdict_ending
            )
        
        content.append({"type": "text", "text": instruction})
        
        # Create messages
        messages = [
            {
                "role": "system",
                "content": "You are a vision-and-language judge. Follow the instructions strictly."
            },
            {
                "role": "user", 
                "content": content
            }
        ]
        
        # Call OpenAI API
        logging.info(f"Calling OpenAI API with model {model_name}")
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        full_response = response.choices[0].message.content.strip()
        
        # Extract and clean final answer
        boxed = extract_final_boxed_content(full_response) or "NONE"
        boxed_clean = clean_answer(boxed)
        
        # Additional cleaning for common formatting issues
        boxed_clean = _strip_outer_braces(boxed_clean)
        boxed_clean = _clean_latex_formatting(boxed_clean)
        
        logging.info(f"[GPT-4o VERDICT OUTPUT ({dataset})]\n{full_response}\n[BOXED]={boxed_clean}")
        
        return full_response, boxed_clean
        
    except Exception as e:
        logging.error(f"GPT-4o API call failed: {e}")
        raise

def qwen_verdict(
    model,                      
    processor,                    
    question: str,
    answers_dict: Dict[str, Dict[str, Any]],   
    orig_img_path: str,  
    *,            
    annotated_folder: str = "",
    dataset: str = "infovqa",
    device: str = "cuda",
) -> Tuple[str, str]:
    """
    Use Qwen model to fuse reasoning from multiple models.
    Uses dataset-specific prompts from prompts.py.
    
    Args:
        model: Loaded Qwen model
        processor: Qwen processor 
        question: Question text
        answers_dict: Dict of {model_tag: {"reasoning": ..., "answer": ...}}
        orig_img_path: Path to original image
        annotated_folder: Optional folder containing layout-annotated images
        dataset: Dataset name for verdict prompt selection
        device: Device for inference
        
    Returns:
        Tuple of (full_response, final_answer)
    """
    # Handle annotated image if folder provided
    has_annotation = False
    anno_img = None
    
    if annotated_folder:
        anno_path = Path(annotated_folder) / f"{Path(orig_img_path).stem}.png"
        if anno_path.exists():
            anno_img = Image.open(anno_path).convert("RGB")
            has_annotation = True
        else:
            print(f"Warning: Annotation not found at {anno_path}")

    # Load original image
    raw_img = Image.open(orig_img_path).convert("RGB")
    
    # Build prompt components
    question_txt = f"Question:\n{question}\n"

    # Format model reasoning blocks
    blocks = []
    for i, (tag, v) in enumerate(answers_dict.items(), start=1):
        reasoning = v.get("reasoning", "")
        ans = v.get("answer", "")
        blocks.append(
            f"--- Model {i} ({tag}) ---\n"
            f"Reasoning:\n{reasoning}\n"
            f"Proposed Answer: {ans}\n"
        )
    blocks_txt = "\n".join(blocks)

    # Get dataset-specific verdict prompt ending from prompts.py
    verdict_ending = get_prompt(task="verdict", dataset=dataset)

    # Combine common prefix with dataset-specific ending
    if has_annotation:
        final_txt = (
            "You are a vision-and-language judge. Follow the instructions strictly. "
            "Given the raw image, the layout-annotated image, the question, and the reasoning from multiple models, "
            + verdict_ending
        )
        prompt_text = question_txt + blocks_txt + final_txt
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": raw_img},
                {"type": "image", "image": anno_img},
                {"type": "text", "text": prompt_text},
            ],
        }]
    else:
        final_txt = (
            "You are a vision-and-language judge. Follow the instructions strictly. "
            "Given the image, the question, and the reasoning from multiple models, "
            + verdict_ending
        )
        prompt_text = question_txt + blocks_txt + final_txt
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": raw_img},
                {"type": "text", "text": prompt_text},
            ],
        }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    gen_kwargs = dict(
        max_new_tokens=2048, 
        temperature=0.01, 
        top_p=0.001, 
        top_k=1, 
        repetition_penalty=1.0
    )
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    boxed = extract_final_boxed_content(output_text)
    boxed = clean_think_tags(boxed)
    boxed_clean = clean_answer(boxed)
    boxed_clean = _strip_outer_braces(boxed_clean)

    annotation_status = "with annotation" if has_annotation else "without annotation"
    print(f"[QWEN VERDICT OUTPUT ({dataset}, {annotation_status})]\n{output_text}\n[BOXED]={boxed_clean}")
    
    return output_text, boxed_clean

def _strip_outer_braces(s: str) -> str:
    """Remove outer braces from answer if present."""
    t = s.strip()
    if t.startswith("{") and t.endswith("}"):
        return t[1:-1].strip()
    return t

def _clean_latex_formatting(text: str) -> str:
    """Clean common LaTeX formatting artifacts."""
    if not text:
        return text
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)  # \cmd{content} -> content
    text = re.sub(r'\\[a-zA-Z]+', '', text)  # Remove standalone commands
    return text.strip()