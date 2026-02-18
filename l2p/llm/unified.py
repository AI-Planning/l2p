"""
Unified LLM wrapper built on top of BaseLLM for multiple providers using 
simonw/llm: https://github.com/simonw/llm. Supports provider-specific 
argument differences internally while keeping a single public class for L2P.

LLM supports: https://llm.datasette.io/en/stable/plugins/directory.html

A YAML configuration file is required to specify model parameters, costs, and other
provider-specific settings. By default, the l2p library includes a configuration file
located at 'l2p/llm/utils/llm.yaml' (different from /openai.yaml due to argument passing).

Users can also define their own custom models and parameters by extending the YAML
configuration using the same format template.
"""

from typing_extensions import override
from .base import BaseLLM, load_yaml

class UnifiedLLM(BaseLLM):
    def __init__(
        self,
        provider: str,
        model: str,
        config_path: str = "l2p/llm/utils/llm.yaml",
        api_key: str | None = None,
    ) -> None:

        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "The 'tiktoken' library is required for token processing but is not installed. "
                "Install it using: `pip install tiktoken`."
            )
        
        try:
            import llm
        except ImportError:
            raise ImportError(
                "The 'llm' library is required for using this unified class. "
                "Install it using: `pip install llm`."
                "To use Ollama: `llm install llm-ollama`"
                "To use DeepSeek: `llm install llm-deepseek`"
                "Visit for more llm supports: https://llm.datasette.io/en/stable/plugins/directory.html"
            )

        self.provider = provider
        self.api_key = api_key
        self._config = load_yaml(config_path)

        if model not in self._config.get(self.provider, {}):
            raise ValueError(
                f"{model} not found under provider {self.provider} in {config_path}. "
                f"Available models: {self.valid_models()}"
            )
        
        model_config = self._config.get(self.provider, {}).get(model, {})

        # model alias and handle
        self.model_alias = model_config.get("model_alias", model)
        self.model_handle = llm.get_model(self.model_alias)

        # tokens and query tracking
        self.tok = tiktoken.get_encoding("cl100k_base")
        self.input_tokens = 0
        self.output_tokens = 0
        self.query_log = []

        # pricing
        self.cost_per_input_token = model_config.get("cost_usd_mtok", {}).get("input", 0)
        self.cost_per_output_token = model_config.get("cost_usd_mtok", {}).get("output", 0)

        # store LLM parameters (provider-specific)
        self.model_params = model_config.get("model_params", {})

    @override
    def query(self, prompt: str, end_when_error=False, max_retry=3, est_margin=200) -> str:
        """Query the LLM w/ prompt using parameters defined in YAML."""
        
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")
        
        # estimate tokens
        current_tokens = len(self.tok.encode(prompt))

        conn_success, n_retry = False, 0
        while not conn_success and n_retry < max_retry:
            try:
                print(
                    f"[INFO] connecting to {self.model_alias} (Prompt estimation: {current_tokens} tokens)..."
                )

                # build kwargs
                kwargs = dict(self.model_params) # all provider-specific params
                kwargs["prompt"] = prompt
                if self.api_key and "key" not in kwargs:
                    kwargs["key"] = self.api_key

                response = self.model_handle.prompt(**kwargs)

                llm_output = response.text()
                if llm_output is None:
                    raise ValueError("LLM returned no output.")
                
                # track usage
                usage = response.usage()
                details = getattr(usage, "details", None)

                if usage:
                    self.input_tokens = usage.input
                    self.output_tokens = usage.output
                else:
                    self.input_tokens = current_tokens
                    self.output_tokens = len(self.tok.encode(llm_output))

                # cost calculation
                input_cost = (self.input_tokens / 1_000_000) * self.cost_per_input_token
                output_cost = (self.output_tokens / 1_000_000) * self.cost_per_output_token
                total_cost = input_cost + output_cost

                self.query_log.append(
                    {
                        "model": self.model_alias,
                        "prompt": prompt,
                        "response": llm_output,
                        "input_tokens": self.input_tokens,
                        "output_tokens": self.output_tokens,
                        "details": details,
                        "input_cost_usd": input_cost,
                        "output_cost_usd": output_cost,
                        "total_cost_usd": total_cost
                    }
                )

                # reset temporary token counts
                self.reset_tokens()
                conn_success = True
            
            except Exception as e:
                print(f"ERROR {e}")
                if end_when_error:
                    break
            n_retry += 1

        if not conn_success:
            raise ConnectionError(f"[ERROR] Failed to connect to LLM after {max_retry} retries.")

        return llm_output
    
    def get_query_log(self) -> list:
        return self.query_log
    
    def reset_query_log(self) -> None:
        self.query_log = []
    
    def get_tokens(self) -> tuple[int, int]:
        return self.input_tokens, self.output_tokens
    
    def reset_tokens(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    @override
    def valid_models(self) -> list[str]:
        """Returns a list of valid model engines."""
        try:
            return list(self._config.get(self.provider, {}).keys())
        except KeyError:
            return []   