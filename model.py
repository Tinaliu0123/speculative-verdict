"""
VLM wrapper for prefilling and generation across multiple backends.

PrefillingModel supports:
1. Prefill NLL computation for consensus scoring
2. Baseline answer generation for draft reasoning

Supported backends:
    Qwen, MiMO, InternVL, GLM, Ovis, Eagle, LLaVA-OneVision, Gemma
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
from torch import Tensor
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info

import torch, json

@dataclass
class Specials:
    """Holds backend-specific chat markers and special token ids."""
    user_prefix: List[int]
    asst_prefix: List[int]
    pad_id: Optional[int] = None
    image_token_ids: Tuple[int, ...] = ()

class PrefillingModel:
    """
    A light wrapper that exposes two capabilities:
      (1) prefill_nll: average NLL on the answer span only (for consensus scoring)
      (2) answer:      baseline generative answering (for draft reasoning)
    """

    def __init__(self, model, processor, tokenizer, tag: str, device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.tag = tag.lower()
        self.device = device

        self.gen_kwargs = self._default_gen_kwargs()
        self.specials = self._specials_for_tag()

    def _default_gen_kwargs(self) -> Dict:
        """
        Returns backend-specific generation defaults. Following each model's own evaluation framework.
        """
        if "qwen" in self.tag:
            return dict(max_new_tokens=2048, temperature=0.01, top_p=0.001, top_k=1, repetition_penalty=1.0)
        if "mimo" in self.tag:
            return dict(max_new_tokens=16384, temperature=0.0, top_p=1.0)
        if "intern" in self.tag:
            return dict(max_new_tokens=4096, do_sample=True, temperature=0.7, top_p=0.95)
        if "glm" in self.tag:
            return dict(max_new_tokens=8192, temperature=0.1, do_sample=True)
        if "ovis" in self.tag:
            return dict(max_new_tokens=3072, do_sample=False, top_p=None, top_k=None, temperature=None, repetition_penalty=None)
        if "llava" in self.tag:
            return dict(max_new_tokens=1024, do_sample=False, temperature=0.0, top_p=None, num_beams = 1)
        if "eagle" in self.tag:
            return dict(do_sample=True, temperature=0.2, top_p=0.5, num_beams=1, max_new_tokens=1024, use_cache=True)
        return dict(max_new_tokens=2048) 

    def _specials_for_tag(self) -> Specials:
        """
        Defines chat markers and special token ids per backend.
        This concentrates all marker/id assumptions in one place.
        """
        if "qwen" in self.tag or "mimo" in self.tag or "intern" in self.tag or "llava" in self.tag:
            user = self.tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
            asst = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            img_ids = (151655, 151652, 151653)
            if "internvl3.5" in self.tag:
                print("3.5")
                img_ids += (151669, 151670, 151671)
            elif "intern" in self.tag:
                img_ids += (151667, 151666, 151665)  # Additional image tokens for InternVL
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            return Specials(user, asst, pad_id, img_ids)

        if "eagle" in self.tag:
            user = self.tokenizer.encode("<|im_start|>user\n<image 1>", add_special_tokens=False)
            asst = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            img_ids = (151655, 151652, 151653, 151667, 151666, 151665)  # Additional image tokens for InternVL
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            return Specials(user, asst, pad_id, img_ids)

        if "glm" in self.tag:
            user = self.tokenizer.encode("<|user|>\n", add_special_tokens=False)
            asst = self.tokenizer.encode("<|assistant|>\n", add_special_tokens=False)
            img_ids = (151343, 151339, 151340)
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            return Specials(user, asst, pad_id, img_ids)

        if "ovis" in self.tag:
            user = self.tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
            asst = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            img_ids = (151655, 151652, 151653)
            pad_id = 151643
            return Specials(user, asst, pad_id, img_ids)
        
        if "gemma" in self.tag:
            user = self.tokenizer.encode("<bos><start_of_turn>user\n\n\n", add_special_tokens=False)
            asst = self.tokenizer.encode("\n<start_of_turn>model\n", add_special_tokens=False)
            img_ids = (255999, 256000, 262144)
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            return Specials(user, asst, pad_id, img_ids)

        return Specials([], [], getattr(self.tokenizer, "pad_token_id", None), ())
        
    def _build_messages(self, image_path: str, text_prompt: str) -> List[Dict]:
        """
        Constructs a single-turn, image-grounded user message.
        InternVL expects a 'url' field; others accept 'image' path.
        """
        if "intern" in self.tag:
            return [{"role": "user",
                     "content": [{"type": "image", "url": image_path},
                                 {"type": "text", "text": text_prompt}]}]

        # For others: Load image as PIL Image object
        img = Image.open(image_path).convert("RGB")    
        return [{"role": "user",
                "content": [{"type": "image", "image": img}, 
                            {"type": "text", "text": text_prompt}]}]
 
    def _apply_template_and_encode(self, messages: List[Dict], image_path: Optional[str] = None) -> Dict[str, Tensor]:
        """
        Applies the backend's chat template and packs multi-modal inputs.
        Returns a dict of tensor inputs on self.device.
        """
        if "glm" in self.tag:
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(self.device)
            return inputs

        if "ovis" in self.tag:
            return self._handle_ovis_encoding(messages)

        if "intern" in self.tag or "gemma" in self.tag:
            inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self.model.device, dtype=torch.bfloat16)
            return inputs

        # Standard processing for qwen, mimo
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.device)
        return inputs

    def _handle_ovis_encoding(self, messages: List[Dict]) -> Dict[str, Tensor]:
        """Special encoding for OVIS model"""
        # Extract image path from messages
        image_path = None
        for content in messages[0]["content"]:
            if content["type"] == "image":
                image_path = content["image"]
                break
        
        if not image_path:
            raise ValueError("No image found in messages for OVIS")

        # OVIS-specific preprocessing
        enable_thinking = False
        min_pixels = 200704
        max_pixels = 2408448

        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        
        input_ids = input_ids.to(self.device)
        
        inputs = {"input_ids": input_ids}
        if pixel_values is not None:
            inputs["pixel_values"] = pixel_values.to(device=self.device, dtype=self.model.dtype)
        if grid_thws is not None:
            inputs["grid_thws"] = grid_thws.to(self.device)
            
        return inputs

    def _eagle_prepare_inputs(self, messages, base_inputs):
        px = base_inputs["pixel_values"]                           
        n_patches = px.shape[0] if px.dim()==4 else px.shape[1]
        n_vit = n_patches * int(self.model.num_image_token)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        ctx = "<img>" + "<IMG_CONTEXT>" * (n_vit + 100) + "</img>"
        for ph in ("<image>", "<image 1>", "<image_1>"):
            if ph in text:
                text = text.replace(ph, ctx, 1)
                break
        else:
            text = ctx + "\n" + text 

        enc = self.tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        img_ctx_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        assert img_ctx_id is not None and img_ctx_id != getattr(self.tokenizer, "unk_token_id", -1), \
            "Tokenizer lacks <IMG_CONTEXT>"
        self.model.img_context_token_id = img_ctx_id

        sel = (input_ids == img_ctx_id)              
        extra = int(sel.sum().item()) - n_vit
        if extra > 0:
            pos = torch.nonzero(sel[0], as_tuple=False)[-extra:] 
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
            input_ids[0, pos[:, 0]] = pad_id
            attention_mask[0, pos[:, 0]] = 0

        image_flags = torch.ones(n_patches, dtype=torch.long, device=px.device)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": px.to(self.device),
            "image_flags": image_flags,
        }

    # Label Masking for Prefill NLL
    def _mask_labels_for_chat(
        self,
        input_ids: torch.Tensor,
        inputs: Dict | None = None,  
        ) -> torch.Tensor:
        """
        Build a `labels` tensor such that *only* the prompt (question + answer) tokens
        contribute to the loss.

        Masking rules
        -------------
        1. Everything up to and including the last <user> prefix.
        2. From the last <assistant> prefix (inclusive) to the end of sequence
        3. PAD tokens and image-placeholder tokens.

        Parameters
        ----------
        input_ids : Tensor
            Shape (1, L). The full tokenized sequence of question and answer tokens.
        inputs : Dict, optional
            Extra inputs returned by the processor; needed only for OVIS.

        Returns
        -------
        Tensor
            A copy of `input_ids` where masked positions are set to −100.
        """
        labels = input_ids.clone()
        ids = input_ids[0]                      

        def mask_by_prefix(prefix_ids: list[int], mask_after: bool, use_first: bool = False) -> None:
            """
            Locate occurrence of `prefix_ids` and apply masking.

            mask_after = False  → mask everything up to and including the prefix.
            mask_after = True   → mask from the prefix start all the way to EOS.
            use_first = True    → use first occurrence (default: last occurrence)
            """
            if not prefix_ids:
                return
            k = len(prefix_ids)
            window = ids.unfold(0, k, 1)        
            pattern = torch.tensor(prefix_ids, device=ids.device)
            hits = (window == pattern).all(dim=1)
            if hits.any():
                idx = 0 if use_first else -1
                pos = int(hits.nonzero(as_tuple=False)[idx, 0])
                
                if mask_after:
                    labels[0, pos:] = -100
                else:
                    labels[0, :pos + k] = -100

        # 1) mask system / user part 
        mask_by_prefix(self.specials.user_prefix, mask_after=False, use_first=True)

        # 2) mask assistant prefix + further output 
        mask_by_prefix(self.specials.asst_prefix, mask_after=True, use_first=False)

        # 3) mask pad and image tokens
        if self.specials.pad_id is not None:
            labels[labels == self.specials.pad_id] = -100
        for tid in self.specials.image_token_ids:
            labels[labels == tid] = -100

        # 4) OVIS: mask negative IDs
        if "ovis" in self.tag and inputs is not None:
            labels[input_ids < 0] = -100
            labels = labels.to(torch.long)
        
        return labels

    @torch.inference_mode()
    def prefill_nll(self, image_path: str, question: str, answer: str) -> float:
        """
        Computes the average negative log-likelihood (NLL) for the provided
        answer text, conditioned on (image, question). Only the answer span
        contributes to the loss via label masking.
        """
        text_prompt = f"Question: {question}\nAnswer: {answer}"
        print(f"Prompt: {text_prompt}")
        
        messages = self._build_messages(image_path, text_prompt)
        inputs = self._apply_template_and_encode(messages, image_path)

        input_ids: Tensor = (inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids).to(self.device)
        labels = self._mask_labels_for_chat(input_ids, inputs)

        # OVIS-specific forward call
        if "ovis" in self.tag:
            attention_mask = torch.ne(input_ids, 151643).to(self.device)
            
            out = self.model(
                input_ids,
                attention_mask=attention_mask,
                pixel_values=inputs["pixel_values"],
                grid_thws=inputs["grid_thws"],
                labels=labels
            )
            avg_nll = out.loss
            ppl = torch.exp(avg_nll)
        elif "eagle" in self.tag:
            import torch.distributed as dist
            if dist.is_available() and not dist.is_initialized():
                dist.get_rank = lambda *a, **k: 0

            packed = self._eagle_prepare_inputs(messages, inputs)

            labels = self._mask_labels_for_chat(packed["input_ids"])

            img_ctx_id = self.model.img_context_token_id
            n_patches = packed["image_flags"].numel()
            n_vit = n_patches * int(self.model.num_image_token)

            self.model.to(self.device).eval()
            avg_nll = self.model(**packed, labels=labels).loss
            ppl = torch.exp(avg_nll)
            print(f"Avg NLL: {avg_nll.item():.4f}\nPPL: {ppl.item():.4f}")
            
            # All other models: pass full inputs dict
        else:
            self.model.to(self.device).eval()
            avg_nll = self.model(**inputs, labels=labels).loss
            ppl = torch.exp(avg_nll)
        
        print(f"Avg NLL: {avg_nll.item():.4f}")
        print(f"PPL: {ppl.item():.4f}")
            
        return ppl.item()

    @torch.inference_mode()
    def answer(self, image_path: str, question: str, prompt_tpl: str,
            return_prompt: bool = False) -> str:
        """
        Generates a baseline answer under a given prompt template.
        """
        prompt = prompt_tpl.format(question)
        print("Answering the questions")
        
        if "ovis" in self.tag:
            return self._answer_ovis(image_path, prompt)
        
        # Standard processing
        messages = self._build_messages(image_path, prompt)
        gen_kwargs = dict(self.gen_kwargs)
        inputs = self._apply_template_and_encode(messages, image_path)
        self.model.to(self.device).eval()
        outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Handle different output formats
        if "glm" in self.tag:
            output_text = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False
            )
        elif "eagle" in self.tag:
            output_text = self.processor.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        else:
            input_ids = getattr(inputs, 'input_ids', None)
            if input_ids is None:
                input_ids = inputs.get('input_ids')
            trimmed = [out[len(inp):] for inp, out in zip(input_ids, outputs)]
            text = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            output_text = text[0]
        
        print(f'User: {prompt}\nAssistant: {output_text}')
        return output_text
   
    def _answer_ovis(self, image_path: str, prompt: str) -> str:
        """Handle ovis-specific answer generation"""
        enable_thinking = False
        enable_thinking_budget = False
        thinking_budget = 2048
        min_pixels = 200704
        max_pixels = 2408448

        msg = [{
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(image_path).convert("RGB")},
                {"type": "text", "text": prompt},
            ],
        }]

        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=msg,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        
        input_ids = input_ids.to(self.device)
        pixel_values = pixel_values.to(self.device) if pixel_values is not None else None
        grid_thws = grid_thws.to(self.device) if grid_thws is not None else None

        outputs = self.model.generate(
            inputs=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            enable_thinking=enable_thinking,
            enable_thinking_budget=enable_thinking_budget,
            thinking_budget=thinking_budget,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **{k: v for k, v in self.gen_kwargs.items()}
        )

        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text