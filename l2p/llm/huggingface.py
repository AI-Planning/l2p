"""
This is a subclass (HUGGING_FACE) for abstract class (BaseLLM) that implements an interface
to interact with downloaded text generation models from the HuggingFace API.

A YAML configuration file is required to specify model parameters and other
provider-specific settings. By default, the l2p library includes a configuration file
located at 'l2p/llm/utils/llm.yaml'.

Users can also define their own custom models and parameters by extending the YAML
configuration using the same format template.
"""

from typing_extensions import override
from .base import BaseLLM, load_yaml
from .utils.prompt_template import prompt_templates
import warnings

warnings.filterwarnings("ignore", message="`do_sample` is set to `False`.*")


class HUGGING_FACE(BaseLLM):
    def __init__(
        self,
        model: str,
        model_path: str,  # base directory of stored model
        config_path: str = "l2p/llm/utils/llm.yaml",
        provider: str = "huggingface",
        api_key: str | None = None,  # only if model is affiliated w/ private repo
    ) -> None:

        # attempt to import neccessary libraries
        try:
            import transformers
            from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

            self.AutoTokenizer = AutoTokenizer
            self.AutoConfig = AutoConfig
            self.AutoModelForCausalLM = AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "The 'transformers' library (and its components like AutoTokenizer) is required for HUGGING_FACE "
                "but is not installed or failed to import properly. Install it using: `pip install transformers`."
            )
        try:
            import torch

            self.torch = torch
        except ImportError:
            raise ImportError(
                "The 'torch' library is required for HUGGING_FACE but is not installed. "
                "Install it using: `pip install torch`."
            )

        self.device = "cuda" if self.torch.cuda.is_available() else "cpu"

        self.api_key = api_key

        # load yaml configuration path
        self.provider = provider
        self._config = load_yaml(config_path)

        # retrieve model configurations
        model_config = self._config.get(self.provider, {}).get(model, {})
        self.model_engine = model_config.get("engine", model)
        self.model_path = model_path

        # check/load model
        self._load_transformer()

        # set parameters for model
        self._set_parameters(model_config)

        # set model configuration
        self._set_configs(model_config)

        # assign other default model parameters
        self.batch_size = 1
        self.pad_token_id = self.tokenizer.eos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        # recording logs
        self.in_tokens = 0
        self.out_tokens = 0
        self.query_log = []

        # set model
        self.llm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_path,
            device_map=self.device_map,
            torch_dtype=self.dtype,
        ).to(self.device)

    def _load_transformer(self):
        """Checks and loads model tokenizer/context length if exists."""

        try:
            # lightweight check — will raise OSError if the model path is invalid
            if self.api_key:
                self.tokenizer = self.AutoTokenizer.from_pretrained(
                    self.model_path, token=self.api_key
                )
            else:
                self.tokenizer = self.AutoTokenizer.from_pretrained(self.model_path)

            self.context_length = self.AutoConfig.from_pretrained(
                self.model_path
            ).max_position_embeddings

        except OSError as e:
            # if model_path is not found, raise an error
            raise ValueError(
                f"Model path '{self.model_path}' could not be found. Please ensure the model exists."
            )
        except Exception as e:
            # catch any other exceptions and raise a generic error
            raise RuntimeError(f"An error occurred while loading the model: {str(e)}")

    def _set_parameters(self, model_config: dict) -> None:
        """Set parameters from the model configuration"""

        # default values for parameters if none exists
        defaults = {
            "context_length": self.context_length,
            "max_new_tokens": 512,
            "temperature": 0.0,
            "top_p": 1.0,
            "stop": None,
            "do_sample": False,
        }

        parameters = model_config.get("model_params", {})
        for key, default in defaults.items():
            setattr(self, key, parameters.get(key, default))

    def _set_configs(self, model_config: dict) -> None:
        """Set model hardware configuration."""

        # Mapping from string to torch.dtype
        dtype_map = {
            "float32": self.torch.float32,
            "float16": self.torch.float16,
            "bfloat16": self.torch.bfloat16,
            "int8": self.torch.int8,
        }

        # Extract inner config if it exists
        configs = model_config.get("model_config", {})

        # Get and parse torch_dtype
        d_type = configs.get("dtype", "float32")
        if isinstance(d_type, str):
            if d_type in dtype_map:
                self.dtype = dtype_map[d_type]
            else:
                raise ValueError(
                    f"Unsupported dtype string: '{d_type}'. Must be one of {list(dtype_map.keys())}."
                )
        elif isinstance(d_type, self.torch.dtype):
            self.dtype = d_type
        else:
            raise TypeError("dtype must be a string or torch.dtype instance.")

        # Set other default config values
        self.device_map = configs.get("device_map", "auto")

    def generate_prompt(self, system_message, prompt):
        """Generate prompt structure for specific LLM."""
        system_message = (
            system_message
            or "You are a PDDL coding assistant. Provide concise, correct code only.\n"
        )
        model_name = self.model_engine.lower()

        # try to find matching template key from the prompt_templates
        for key in prompt_templates:
            if key in model_name:
                template = prompt_templates[key]
                return template.format(
                    system_prompt=system_message, prompt=prompt
                ).strip()

        # default fallback if no template matched
        return prompt

    @override
    def query(
        self,
        prompt: str,
        system_prompt: str = None,
        end_when_error: bool = False,
        max_retry: int = 3,
        est_margin: int = 200,
    ) -> str:
        """Generate a response from HuggingFace model based on the prompt."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        full_prompt = self.generate_prompt(system_prompt, prompt)
        assert full_prompt is not None

        input = self.tokenizer([full_prompt], return_tensors="pt")
        requested_tokens = len(input.input_ids[0])

        available_context = self.context_length - requested_tokens - est_margin
        max_new_tokens = min(self.max_new_tokens, max(0, available_context))

        print(
            f"Requesting {max_new_tokens} tokens "
            f"(estimated prompt: {requested_tokens} tokens, margin: {est_margin}, window: {self.context_length})"
        )

        # request response
        conn_success, n_retry = False, 0
        while not conn_success and n_retry < max_retry:
            try:
                # print token information
                print(
                    f"[INFO] connecting to {self.model_engine} ({requested_tokens} tokens)..."
                )
                if requested_tokens >= self.context_length:
                    print(
                        f"[WARNING] Prompt is {requested_tokens} tokens and exceeds context length "
                        f"({self.context_length}). It will be truncated."
                    )

                input = {k: v.to(self.device) for k, v in input.items()}

                # get response from LLM
                with self.torch.no_grad():
                    outputs = self.llm.generate(
                        **input,
                        max_new_tokens=max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        pad_token_id=self.pad_token_id,
                        eos_token_id=self.eos_token_id,
                        do_sample=self.do_sample,
                    )

                # retrieve output content from LLM response
                llm_output = self.tokenizer.decode(
                    outputs[0][requested_tokens:], skip_special_tokens=True
                )

                # exclude texts after stop token
                if self.stop is not None:
                    llm_output = llm_output.split(self.stop)[0]

                conn_success = True

            except Exception as e:
                print(f"[ERROR] LLM error: {e}")
                if end_when_error:
                    break
                n_retry += 1

        if not conn_success:
            raise ConnectionError(
                f"Failed to generate response after {max_retry} retries."
            )

        # retrieve output tokens
        output_ids = self.tokenizer(
            llm_output, return_tensors="pt", truncation=True
        ).input_ids
        output_token_count = len(output_ids[0])

        # record token counts
        self.in_tokens += requested_tokens
        self.out_tokens += output_token_count

        self.query_log.append(
            {
                "model": self.model_engine,
                "prompt_tokens": requested_tokens,
                "completion_tokens": output_token_count,
                "total_tokens": requested_tokens + output_token_count,
                "prompt": full_prompt,
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
