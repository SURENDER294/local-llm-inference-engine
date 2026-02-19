import time
import sys
from typing import Optional, Iterator, List

import torch
import torch.nn.functional as F

from .model_loader import load_model
from .tokenizer import BPETokenizer
from .kv_cache import KVCache
from .sampler import Sampler
from .quantizer import quantize_int8


class InferenceEngine:
    """
    Main entry point for running local LLM inference.

    Handles model loading, tokenization, KV-cache management,
    and token sampling. No external API calls.
    """

    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        device: str = "cpu",
        use_int8: bool = False,
        max_seq_len: int = 2048,
    ):
        """
        Args:
            model_path: Path to .safetensors or .bin weights file.
            vocab_path: Path to tokenizer vocabulary (vocab.json + merges.txt).
            device: 'cpu' or 'cuda'.
            use_int8: Apply INT8 quantization to reduce memory usage.
            max_seq_len: Maximum context length.
        """
        self.device = torch.device(device)
        self.max_seq_len = max_seq_len

        print(f"[engine] Loading tokenizer from {vocab_path}")
        self.tokenizer = BPETokenizer(vocab_path)

        print(f"[engine] Loading model from {model_path}")
        self.model = load_model(model_path, device=device)

        if use_int8:
            print("[engine] Applying INT8 quantization...")
            self.model = quantize_int8(self.model)

        self.model.eval()
        self.sampler = Sampler()
        print(f"[engine] Ready on {device}.")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        stop_token: Optional[str] = None,
    ) -> str:
        """
        Generate text for a given prompt.

        Args:
            prompt: Input text string.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (1.0 = neutral).
            top_k: Keep only top-k logits before sampling.
            top_p: Nucleus sampling probability threshold.
            stop_token: Optional stop string to end generation early.

        Returns:
            Generated text (not including the prompt).
        """
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        kv_cache = KVCache(max_seq_len=self.max_seq_len)
        generated_ids: List[int] = []

        for _ in range(max_new_tokens):
            # Forward pass — pass only the new token(s) after first step
            if kv_cache.size == 0:
                logits = self.model(input_ids, kv_cache=kv_cache)
            else:
                last_token = torch.tensor(
                    [[generated_ids[-1]]], dtype=torch.long, device=self.device
                )
                logits = self.model(last_token, kv_cache=kv_cache)

            # Sample next token
            next_token_id = self.sampler.sample(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            generated_ids.append(next_token_id)

            # Check for EOS or stop token
            if next_token_id == self.tokenizer.eos_token_id:
                break

            if stop_token:
                partial = self.tokenizer.decode(generated_ids)
                if stop_token in partial:
                    break

        return self.tokenizer.decode(generated_ids)

    @torch.no_grad()
    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> Iterator[str]:
        """
        Stream tokens one at a time to stdout (like ChatGPT typing effect).

        Yields each decoded token string as it’s generated.
        """
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        kv_cache = KVCache(max_seq_len=self.max_seq_len)
        generated_ids: List[int] = []

        for _ in range(max_new_tokens):
            if kv_cache.size == 0:
                logits = self.model(input_ids, kv_cache=kv_cache)
            else:
                last_token = torch.tensor(
                    [[generated_ids[-1]]], dtype=torch.long, device=self.device
                )
                logits = self.model(last_token, kv_cache=kv_cache)

            next_token_id = self.sampler.sample(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            generated_ids.append(next_token_id)

            # Decode just the new token and yield it
            token_text = self.tokenizer.decode([next_token_id])
            yield token_text

            if next_token_id == self.tokenizer.eos_token_id:
                break

    def __repr__(self) -> str:
        return (
            f"InferenceEngine(device={self.device}, "
            f"max_seq_len={self.max_seq_len})"
        )
