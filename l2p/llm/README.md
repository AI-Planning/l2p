# L2P LLM — Multi-Provider Language Model Interface

> Unified abstraction over LLM backends for PDDL generation.

```
l2p/llm/
├── base.py          # Abstract BaseLLM + require_llm decorator
├── openai.py        # OpenAI SDK provider (OpenAI, DeepSeek, Anthropic, etc.)
├── unified.py       # simonw/llm provider (Ollama, plus all llm plugins)
├── huggingface.py   # Local HuggingFace transformers
├── vllm.py          # vLLM accelerated inference (experimental)
└── utils/
    ├── llm.yaml         # UnifiedLLM model config
    ├── openaiSDK.yaml   # OPENAI model config
    └── prompt_templates.py  # Chat template formatters
```

---

## Base Interface

Every provider extends `BaseLLM` and must implement `query(prompt) -> str`:

```python
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    def __init__(self, model: str, api_key: str | None = None)
    @abstractmethod
    def query(self, prompt: str) -> str: ...
    def query_with_system_prompt(self, system: str, prompt: str) -> str: ...
    def valid_models(self) -> list[str]: ...
```

The `@require_llm` decorator (from `base.py`) validates that a `BaseLLM` instance is passed to builder methods, providing clear errors when missing.

All provider classes track token usage, cost, and query history:

```python
llm.get_tokens()      # -> (input_tokens, output_tokens)
llm.reset_tokens()
llm.get_query_log()   # -> list of per-query metadata dicts
llm.reset_query_log()
```

---

## `OPENAI` — OpenAI SDK Backend

Compatible with any provider that implements the OpenAI chat completions API format:

- OpenAI (`gpt-4o`, `gpt-4o-mini`, `o1-mini`, `gpt-5`, ...)
- DeepSeek (`deepseek-chat`, `deepseek-reasoner`)
- Anthropic via OpenRouter or custom endpoints
- Ollama Cloud
- Any OpenAI-compatible endpoint

```python
from l2p.llm.openai import OPENAI

llm = OPENAI(
    provider="openai",            # or "deepseek", "anthropic", etc.
    model="gpt-4o-mini",
    config_path="l2p/llm/utils/openaiSDK.yaml",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1/",  # optional override
)

response = llm.query("Generate predicates for blocksworld.")
```

Configuration is driven by `openaiSDK.yaml`:

```yaml
openai:
  base_url: "https://api.openai.com/v1/"
  gpt-4o-mini:
    model_alias: "gpt-4o-mini"
    model_family: "gpt"
    model_context_length: 128000
    cost_usd_mtok:
      input: 0.15
      output: 0.60
    model_params:
      max_completion_tokens: 4096
      temperature: 0.0
```

Key features:
- **Automatic token estimation** with margin safety
- **Retry logic** (60s delay between attempts)
- **Cost tracking** per query (USD)
- **Supports Mistral** via its native client library
- Custom `base_url` for proxy/self-hosted endpoints

---

## `UnifiedLLM` — simonw/llm Backend

Wraps the [`llm` library](https://llm.datasette.io), supporting all providers available through its plugin system:

- OpenAI, Google Gemini, Anthropic Claude
- **Ollama** (local models) via `llm install llm-ollama`
- DeepSeek, Mistral, and many more via plugins

```python
from l2p.llm.unified import UnifiedLLM

llm = UnifiedLLM(
    provider="ollama",
    model="llama3.1:8b",
    config_path="l2p/llm/utils/llm.yaml",
)
response = llm.query("Generate PDDL types for a rover domain.")
```

Configuration from `llm.yaml`:

```yaml
ollama:
  llama3.1:8b:
    model_alias: "ollama/llama3.1:8b"
    model_params:
      max_tokens: 4096
      temperature: 0.0
```

The `model_alias` field follows the `llm` plugin convention (e.g., `ollama/llama3.1:8b`, `deepseek/deepseek-chat`).

---

## `HUGGING_FACE` — Local Transformers

Run models locally via HuggingFace `transformers` with automatic prompt formatting:

```python
from l2p.llm.huggingface import HUGGING_FACE

llm = HUGGING_FACE(
    model="llama-2-7b",
    model_path="/path/to/model/files",
    provider="huggingface",
)
response = llm.query("Generate PDDL predicates.")
```

- Auto-detects chat template from model name (Llama, Mistral, Gemma, etc.)
- Configurable `device_map`, `dtype` (float16, bfloat16), `tensor_parallel`
- Tracks token usage

---

## `VLLM` — Accelerated Inference (Experimental)

High-throughput local inference using vLLM:

```python
from l2p.llm.vllm import VLLM

llm = VLLM(
    model="llama-2-7b",
    model_path="/path/to/model",
    provider="huggingface",
)
```

Supports `tensor_parallel_size` for multi-GPU and configurable `gpu_memory_utilization`.

---

## Configuration Files

Two YAML files ship with the library:

| File | Used By | Format |
|------|---------|--------|
| `l2p/llm/utils/llm.yaml` | `UnifiedLLM`, `HUGGING_FACE`, `VLLM` | Per-provider model lists |
| `l2p/llm/utils/openaiSDK.yaml` | `OPENAI` | Per-provider model lists + base_url |

Users can define custom providers/models by adding entries to these files or pointing to their own YAML via `config_path`.

Each entry supports:
- `model_alias` — actual name passed to the API
- `model_context_length` — max tokens window
- `cost_usd_mtok` — input/output pricing (for cost tracking)
- `model_params` — generation parameters (temperature, max_tokens, etc.)
- `model_config` — hardware settings (dtype, device_map, ngpu)

---

## Adding a New Provider

1. Create a subclass of `BaseLLM`
2. Implement `query()` and `valid_models()`
3. Add model entries to the relevant YAML config
4. Optionally register in `l2p/llm/__init__.py`

The abstraction is designed so that every L2P builder works with any provider with zero code changes.
