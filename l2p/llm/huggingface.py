"""
This is a subclass (HUGGING_FACE) for abstract class (BaseLLM) that implements an interface 
to interact with downloaded text generation models from the HuggingFace API. 

A YAML configuration file is required to specify model parameters, costs, and other 
provider-specific settings. By default, the l2p library includes a configuration file 
located at 'l2p/llm/llm.yaml'.

Users can also define their own custom models and parameters by extending the YAML 
configuration using the same format template.
"""

from retry import retry
from typing_extensions import override
from .base import BaseLLM, load_yaml

class HUGGING_FACE(BaseLLM):
    def __init__(
            self, 
            model: str,
            model_path: str, # base directory of stored model
            config_path: str = "l2p/llm/llm.yaml",
            provider: str = "huggingface"
        ) -> None:
        
        # load yaml configuration path
        self.provider = provider
        self._config = load_yaml(config_path)
        
        model_config = self._config.get(self.provider, {}).get(model, {})
        self.model_name = model_config.get("engine", model)
        
        
        
        self.model_path = model_path
        self._load_transformers()

    def _load_transformers(self):

        # attempt to import the `transformers` library
        try:
            import transformers
        except ImportError:
            raise ImportError(
                "The 'transformers' library is required for HUGGING_FACE but is not installed. "
                "Install it using: `pip install transformers`."
            )

        # attempt to import `AutoTokenizer` from `transformers`
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "The 'transformers.AutoTokenizer' module is required but is not installed properly. "
                "Ensure that the 'transformers' library is installed correctly."
            )

        # attempt to import the `torch` library
        try:
            import torch
        except ImportError:
            raise ImportError(
                "The 'torch' library is required for HUGGING_FACE but is not installed. "
                "Install it using: `pip install torch`."
            )

        try:
            # Check if the model_path is valid by trying to load it
            self.model = transformers.pipeline(
                "text-generation",
                model=self.model_path,
                model_kwargs={"torch_dtype": "auto"},
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
        except OSError as e:
            # if model_path is not found, raise an error
            raise ValueError(
                f"Model path '{self.model_path}' could not be found. Please ensure the model exists."
            )
        except Exception as e:
            # catch any other exceptions and raise a generic error
            raise RuntimeError(f"An error occurred while loading the model: {str(e)}")

    # Retry decorator to handle retries on request
    @retry(tries=2, delay=60)
    def connect_huggingface(self, input, temperature, max_tokens, top_p, numSample):
        if self.model is None or self.tokenizer is None:
            self._load_transformers()

        if numSample > 1:
            responses = []
            sequences = self.model(
                input,
                do_sample=True,
                top_k=1,
                num_return_sequences=numSample,
                max_new_tokens=max_tokens,
                return_full_text=False,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            for seq in sequences:
                response = seq["generated_text"]
                responses.append(response)
            return responses

        else:
            sequences = self.model(
                input,
                do_sample=True,
                num_return_sequences=1,
                max_new_tokens=max_tokens,
                return_full_text=False,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            seq = sequences[0]
            response = seq["generated_text"]

            return response

    @override
    def query(self, prompt: str, numSample=1, max_retry=3, est_margin=200) -> str:
        if prompt is None:
            raise ValueError("Prompt cannot be None")

        # Estimate current usage of tokens
        current_tokens = len(self.tokenizer.encode(prompt))
        requested_tokens = min(
            self.max_tokens, self.max_tokens - current_tokens - est_margin
        )

        print(
            f"Requesting {requested_tokens} tokens from {self.model} (estimated {current_tokens - est_margin} prompt tokens with a safety margin of {est_margin} tokens)"
        )

        # Retry logic for Hugging Face request
        n_retry = 0
        conn_success = False
        while not conn_success and n_retry < max_retry:
            n_retry += 1
            try:
                print(
                    f"[INFO] Connecting to Hugging Face model ({requested_tokens} tokens)..."
                )
                llm_output = self.connect_huggingface(
                    input=prompt,
                    temperature=self.temperature,
                    max_tokens=requested_tokens,
                    top_p=self.top_p,
                    numSample=numSample,
                )
                conn_success = True
            except Exception as e:
                print(f"[ERROR] Hugging Face error: {e}")
                if n_retry >= max_retry:
                    raise ConnectionError(
                        f"Failed to connect to the Hugging Face model after {max_retry} retries"
                    )

        # Token management
        response_tokens = len(self.tokenizer.encode(llm_output))
        self.out_tokens += response_tokens
        self.in_tokens += current_tokens

        return llm_output

    def get_tokens(self) -> tuple[int, int]:
        return self.in_tokens, self.out_tokens

    def reset_tokens(self) -> None:
        self.in_tokens = 0
        self.out_tokens = 0
