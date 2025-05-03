"""
This is a subclass (HUGGING_FACE) for abstract class (BaseLLM) that implements an interface 
to interact with downloaded text generation models from the HuggingFace API. 

A YAML configuration file is required to specify model parameters, costs, and other 
provider-specific settings. By default, the l2p library includes a configuration file 
located at 'l2p/llm/utils/llm.yaml'.

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
            config_path: str = "l2p/llm/utils/llm.yaml",
            provider: str = "huggingface"
        ) -> None:
        
        # load yaml configuration path
        self.provider = provider
        self._config = load_yaml(config_path)
        
        # retrieve model configurations
        model_config = self._config.get(self.provider, {}).get(model, {})
        self.model_engine = model_config.get("engine", model)
        self.model_path = model_path

        # load model
        self._load_transformers()

        self._set_parameters(model_config)

        print(self.context_length)
        print(self.temperature)
        print(self.top_p)


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
            from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
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
                model=self.model_engine,
                model_kwargs={"torch_dtype": "float16"},
                device_map="auto",
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_engine)
            self.context_length = AutoConfig.from_pretrained(self.model_engine).max_position_embeddings
            
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

        # default values for paramters if none exists
        defaults = {
            "context_length": self.context_length,
            "temperature": 0.0,
            "top_p": 1.0,
            "stop": None
        }

        parameters = model_config.get("model_params", {})
        for key, default in defaults.items():
            setattr(self, key, parameters.get(key, default))

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
                do_sample=False,
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
    def query(
        self, 
        prompt: str, 
        numSample=1, 
        end_when_error=False, 
        max_retry=3, 
        est_margin=200
        ) -> str:
        """Generate a response from HuggingFace based on prompt."""

        if prompt is None:
            raise ValueError("Prompt cannot be None")

        # estimate current usage of tokens
        current_tokens = len(self.tokenizer.encode(prompt))
        requested_tokens = min(self.context_length, self.context_length - current_tokens - est_margin)

        print(
            f"Requesting {requested_tokens} tokens "
            f"(estimated prompt: {current_tokens} tokens, margin: {est_margin}, window: {self.context_length}"
        )



        # # request response
        # conn_success, n_retry = False, 0
        # while not conn_success and n_retry < max_retry:
        #     try:
        #         print(f"[INFO] Connecting to {self.model_engine} ({requested_tokens} tokens)...")

        #         kwargs = {
        #             "temperature":,
        #             "max_new_tokens":,
        #             "top_p":,
        #             "stop":,
        #             "do_sample":,
                    
        #         }

        #         llm_output = self.connect_huggingface(
        #             input=prompt,
        #             temperature=self.temperature,
        #             max_tokens=requested_tokens,
        #             top_p=self.top_p,
        #             numSample=numSample,
        #         )
        #         conn_success = True
        #     except Exception as e:
        #         print(f"[ERROR] LLM error: {e}")
        #         if end_when_error:
        #             break
            
        #     n_retry += 1

        # # Token management
        # response_tokens = len(self.tokenizer.encode(llm_output))
        # self.out_tokens += response_tokens
        # self.in_tokens += current_tokens

        # return llm_output

    def get_tokens(self) -> tuple[int, int]:
        return self.in_tokens, self.out_tokens

    def reset_tokens(self) -> None:
        self.in_tokens = 0
        self.out_tokens = 0

    @override
    def valid_models(self) -> list[str]:
        """Return a list of valid model names."""
        try:
            return list(self._config.get(self.provider, {}).keys())
        except KeyError:
            return []

if __name__ == "__main__":

    llm = HUGGING_FACE(model="gpt2", model_path="/Users/marcustantakoun/.cache/huggingface/hub/models--openai-community--gpt2")
    
