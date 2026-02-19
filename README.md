# Local LLM Inference Engine

![python](https://img.shields.io/badge/python-3.8+-blue) ![license](https://img.shields.io/badge/license-MIT-green) ![no-api](https://img.shields.io/badge/API_keys-none_required-brightgreen)

> Run open-weight LLMs (GPT-2, TinyLlama, Mistral-7B) entirely on your machine — no cloud, no API keys. Pure Python 3.8 with NumPy + PyTorch inference.

## What This Builds

Most inference tutorials call `model.generate()` and move on. This repo implements the full stack:

- **Tokenizer** — BPE tokenization from a saved vocabulary file
- **Model loader** — load `.safetensors` / `.bin` weights from HuggingFace Hub offline
- **KV-Cache** — past key/value attention cache for fast autoregressive decoding
- **Sampling strategies** — greedy, top-k, top-p (nucleus), temperature scaling
- **Quantization** — INT8 weight quantization to cut VRAM in half
- **Batched inference** — process multiple prompts simultaneously
- **Streaming output** — token-by-token stdout streaming like ChatGPT

## Architecture

```
local_llm/
├── tokenizer.py      # BPE tokenizer with vocab loading
├── model_loader.py   # Load weights from disk (.safetensors)
├── kv_cache.py       # Attention key-value cache
├── sampler.py        # Greedy / top-k / top-p / temperature
├── quantizer.py      # INT8 dynamic quantization
├── engine.py         # Main inference engine (orchestrator)
└── streamer.py       # Token-by-token streaming to stdout
```

## Quick Start

```bash
git clone https://github.com/SURENDER294/local-llm-inference-engine
cd local-llm-inference-engine
pip install -r requirements.txt

# Download a model (one-time, no API key needed)
python scripts/download_model.py --model gpt2

# Run inference
python run.py --model gpt2 --prompt "The future of AI is" --max_tokens 100
```

## Key Features

| Feature | Details |
|---|---|
| Models supported | GPT-2, TinyLlama-1.1B, Mistral-7B-Instruct |
| Quantization | INT8 (2x memory reduction) |
| Sampling | greedy, top-k, top-p, temperature |
| KV-Cache | reduces decode latency by ~3x |
| Batch size | configurable, CPU + CUDA |

## Why No API Keys?

All models run from local weights via `torch.load` or `safetensors`. No OpenAI, no Anthropic, no external calls. Great for air-gapped environments or cost control.

## Stack

- Python 3.8
- PyTorch (CPU or CUDA)
- safetensors
- numpy
- transformers (tokenizer only, no generation)
