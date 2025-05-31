"""
This is a subclass (OPENAI) for abstract class (BaseLLM) that implements an interface
to interact with OpenAI's chat-based language models via the official OpenAI API. It
supports providers that are compatible with OpenAI SDK usage (i.e. DeepSeek, Anthropic, etc.).

A YAML configuration file is required to specify model parameters, costs, and other
provider-specific settings. By default, the l2p library includes a configuration file
located at 'l2p/llm/utils/llm.yaml'.

Users can also define their own custom models and parameters by extending the YAML
configuration using the same format template.
"""

from retry import retry
from typing_extensions import override
from .base import BaseLLM, load_yaml


class OPENAI(BaseLLM):
    def __init__(
        self,
        model: str,
        config_path: str = "l2p/llm/utils/llm.yaml",
        provider: str = "openai",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1/",
    ) -> None:

        # load yaml configuration path
        self.provider = provider
        self._config = load_yaml(config_path)

        # attempt to import necessary OPENAI modules
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' library is required for OPENAI but is not installed. "
                "Install it using: `pip install openai`."
            )

        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "The 'tiktoken' library is required for token processing but is not installed. "
                "Install it using: `pip install tiktoken`."
            )

        # retrieve model configurations
        model_config = self._config.get(self.provider, {}).get(model, {})
        self.model_engine = model_config.get("engine", model)

        # call the parent class constructor to handle model and api_key
        super().__init__(model, api_key)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        # set model parameters
        self._set_parameters(model_config)

        # initialize tokenizer and metadata storage
        self.tok = tiktoken.get_encoding("cl100k_base")
        self.in_tokens = 0
        self.out_tokens = 0
        self.query_log = []  # per-query metadata storage

        # Retrieve cost information for the model from the YAML
        self.cost_per_input_token = model_config.get("cost_usd_mtok", {}).get(
            "input", 0
        )
        self.cost_per_output_token = model_config.get("cost_usd_mtok", {}).get(
            "output", 0
        )

    def _set_parameters(self, model_config: dict) -> None:
        """Set parameters from the model configuration."""

        # default values for parameters if none exists
        defaults = {
            "context_length": 4096,
            "max_completion_tokens": 4096,
            "temperature": 0.0,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": None,
            "reasoning_effort": None,
        }

        parameters = model_config.get("model_params", {})
        for key, default in defaults.items():
            setattr(self, key, parameters.get(key, default))

    @retry(tries=2, delay=60)
    def connect_openai(self, client, model, messages, **kwargs):
        """Send a request to OpenAI API"""

        return client.chat.completions.create(model=model, messages=messages, **kwargs)

    @override
    def query(
        self,
        prompt: str,
        messages=None,
        end_when_error=False,
        max_retry=3,
        est_margin=200,
    ) -> str:
        """Generate a response from OpenAI based on the prompt."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        messages = messages or [{"role": "user", "content": prompt}]

        # estimate current usage of tokens
        current_tokens = sum(len(self.tok.encode(m["content"])) for m in messages)
        requested_tokens = min(
            self.max_completion_tokens,
            self.context_length - current_tokens - est_margin,
        )

        print(
            f"Requesting {requested_tokens} tokens "
            f"(estimated prompt: {current_tokens} tokens, margin: {est_margin}, window: {self.context_length})"
        )

        # request response
        conn_success, n_retry = False, 0
        while not conn_success and n_retry < max_retry:
            try:
                print(
                    f"[INFO] connecting to {self.model_engine} ({requested_tokens} tokens)..."
                )

                kwargs = {
                    "temperature": self.temperature,
                    "max_completion_tokens": requested_tokens,
                    "top_p": self.top_p,
                    "frequency_penalty": self.frequency_penalty,
                    "presence_penalty": self.presence_penalty,
                    "stop": self.stop,
                }

                # only add reasoning_effort if it is not None
                if self.reasoning_effort is not None:
                    kwargs["reasoning_effort"] = self.reasoning_effort

                # retrieve completion
                response = self.connect_openai(
                    client=self.client,
                    model=self.model_engine,
                    messages=messages,
                    **kwargs,
                )

                llm_output = response.choices[
                    0
                ].message.content  # retrieve output from completion

                # record token usage
                usage = getattr(response, "usage", None)
                if usage:
                    self.in_tokens += usage.prompt_tokens
                    self.out_tokens += usage.completion_tokens
                else:
                    self.in_tokens += current_tokens
                    self.out_tokens += len(self.tok.encode(llm_output))

                # calculate cost (USD) per million tokens
                input_cost = (self.in_tokens / 1_000_000) * self.cost_per_input_token
                output_cost = (self.out_tokens / 1_000_000) * self.cost_per_output_token
                total_cost = input_cost + output_cost

                conn_success = True

            except Exception as e:
                print(f"[ERROR] LLM error: {e}")
                if end_when_error:
                    break

            n_retry += 1

        if not conn_success:
            raise ConnectionError(
                f"Failed to connect to the LLM after {max_retry} retries"
            )

        # log query information
        self.query_log.append(
            {
                "model": self.model_engine,
                "prompt_tokens": usage.prompt_tokens if usage else current_tokens,
                "completion_tokens": (
                    usage.completion_tokens
                    if usage
                    else len(self.tok.encode(llm_output))
                ),
                "reasoning_tokens": (
                    usage.completion_tokens_details.reasoning_tokens if usage else 0
                ),
                "total_tokens": (
                    usage.total_tokens
                    if usage
                    else current_tokens + len(self.tok.encode(llm_output))
                ),
                "input_cost_usd": input_cost,
                "output_cost_usd": output_cost,
                "total_cost_usd": total_cost,
                "messages": messages,
                "output": llm_output,
            }
        )

        return llm_output

    def get_tokens(self) -> tuple[int, int]:
        """Return input and output token counts."""
        return self.in_tokens, self.out_tokens

    def reset_tokens(self) -> None:
        """Reset token counts."""
        self.in_tokens = 0
        self.out_tokens = 0

    def get_query_log(self) -> list:
        """Retrieve query log."""
        return self.query_log

    def reset_query_log(self) -> None:
        """Reset query log."""
        self.query_log = []

    @override
    def valid_models(self) -> list[str]:
        """Returns a list of valid model engines."""
        try:
            return list(self._config.get(self.provider, {}).keys())
        except KeyError:
            return []
