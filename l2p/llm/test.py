from .base import BaseLLM, load_yaml
from .utils.prompt_template import prompt_templates
from retry import retry
from typing_extensions import override

class HUGGING_FACE_TEST(BaseLLM):
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
        
        # check/load model
        self._load_transformer()
        
        # set parameters for model
        self._set_parameters(model_config)
        
        # set model configuration
        self._set_configs(model_config)
        
        self.model = model
        self.batch_size = 1
        self.pad_token_id = self.tokenizer.eos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        
        self.llm = AutoModelForCausalLM.from_pretrained(model,
                                                        device_map=self.device_map,
                                                        torch_dtype=self.torch_dtype)
        
        print(self.model)
        print(self.max_tokens)
        print(self.temperature)
        print(self.top_p)
        print(self.stop)
        print(self.torch_dtype)
        print(self.device_map)
        
    def _load_transformer(self):

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
            # lightweight check — will raise OSError if the model path is invalid
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
        
        # default values for parameters if none exists
        defaults = {
            "context_length": self.context_length,
            "max_tokens": 1024,
            "temperature": 0.0,
            "top_p": 1.0,
            "stop": None
        }
        
        parameters = model_config.get("model_params", {})
        for key, default in defaults.items():
            setattr(self, key, parameters.get(key, default))
            
    def _set_configs(self, model_config: dict) -> None:
        """Set model configuration."""

        # Mapping from string to torch.dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int8": torch.int8,
        }

        # Extract inner config if it exists
        configs = model_config.get("model_config", {})

        # Get and parse torch_dtype
        d_type = configs.get("torch_dtype", "float32")
        if isinstance(d_type, str):
            if d_type in dtype_map:
                self.torch_dtype = dtype_map[d_type]
            else:
                raise ValueError(f"Unsupported torch_dtype string: '{d_type}'. Must be one of {list(dtype_map.keys())}.")
        elif isinstance(d_type, torch.dtype):
            self.torch_dtype = d_type
        else:
            raise TypeError("torch_dtype must be a string or torch.dtype instance.")

        # Set other default config values
        self.device_map = configs.get("device_map", "auto")
        
    
    def generate_prompt(self, prompt):
        full_prompt = None
        system_message += "You are a PDDL coding assistant. Provide concise, correct code only. \n"
        
        if "codellama-13b" in self.model.lower():
            full_prompt = prompt_templates["codellama-13b"].format(system_prompt=system_message, prompt=prompt)
            full_prompt = full_prompt.strip()
        elif "codellama-34b" in self.model.lower():
            full_prompt = prompt_templates["codellama-34b"].format(system_prompt=system_message, prompt=prompt)
            full_prompt = full_prompt.strip()
            
        elif "llama" in self.model.lower():
            full_prompt = prompt_templates["llama"].format(system_prompt=system_message, prompt=prompt)
            full_prompt = full_prompt.strip()
            
        elif 'lemur' in self.model.lower():
            full_prompt = prompt_templates["lemur"].format(system_prompt=system_message, prompt=prompt)
            full_prompt = full_prompt.strip()
            
        elif 'vicuna' in self.model.lower():
            full_prompt = prompt_templates["vicuna"].format(system_prompt=system_message, prompt=prompt)
            full_prompt = full_prompt.strip()
        else:
            raise NotImplementedError
        
        return full_prompt
    
    @override
    def query(
        self, 
        prompt: str,
        end_when_error: bool=False,
        max_retry: int=3,
        est_margin: int=200,
        ) -> str:
        """Generate a response from HuggingFace model based on the prompt."""
        
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")
        
        full_prompt = self.generate_prompt(prompt)
        assert full_prompt is not None
        
        with torch.no_grad():
            input = self.tokenizer([full_prompt], return_tensors="pt")
            prompt_length = len(input.input_ids[0])
            input = {k: v.to("cuda") for k, v in input.items()}
            
            outputs = self.llm.generate(**input,
                                        max_new_tokens = self.max_tokens,
                                        temperature = self.temperature,
                                        top_p = self.top_p,
                                        pad_token_id = self.pad_token_id,
                                        eos_token_id = self.eos_token_id,
                                        output_scores = True,
                                        do_sample = False)
            
            output_texts = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
            
            if self.stop is not None:
                output_texts = output_texts.split(self.stop)[0]
            return True, output_texts
    
        
if __name__ == "__main__":
    
    model = "gpt2"
    HUGGING_FACE_TEST(model=model)