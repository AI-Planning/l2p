# TODO - implement vLLM provider for faster inference

"""
This is a subclass (VLLM) for abstract class (BaseLLM) that implements an interface 
to interact with downloaded text generation models (compatible with HuggingFace models). 

A YAML configuration file is required to specify model parameters and other 
provider-specific settings. By default, the l2p library includes a configuration file 
located at 'l2p/llm/utils/llm.yaml'.

Users can also define their own custom models and parameters by extending the YAML 
configuration using the same format template.

There are two variations that users can use vLLM:
    1. Native vLLM
    2. OpenAI-Compatible Server (currently not supported by l2p)
"""

from typing_extensions import override
from .base import BaseLLM, load_yaml
from .utils.prompt_template import prompt_templates

class VLLM(BaseLLM):
    def __init__(
            self, 
            model, 
            model_path: str, # base directory of stored model
            config_path: str = "l2p/llm/utils/llm.yaml",
            provider: str = "huggingface",
            api_key: str | None = None, # only if model is affiliated w/ private repo
        ) -> None:

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "The 'vllm' library is required for VLLM but is not installed or "
                "failed to import properly. Install it using: `pip install vllm`."
            )
        
        self.api_key = api_key
        
        # load yaml configuration path
        self.provider = provider
        self._config = load_yaml(config_path)
            
        # retrieve model configurations
        model_config = self._config.get(self.provider, {}).get(model, {})
        self.model_engine = model_config.get("engine", model)
        self.model_path = model_path

        # set parameters for model
        self._set_parameters(model_config)

        # set model configurations
        self._set_configs(model_config)

        self.sampling_params = SamplingParams(
            temperature = self.temperature,
            top_p = self.top_p,
            stop = self.stop,
            max_tokens = self.max_new_tokens,
        )

        # recording logs
        self.in_tokens = 0
        self.out_tokens = 0
        self.query_log = []

        if self.context_length > 8192:
            self.llm = LLM(
                model = self.model_path,
                dtype = self.dtype,
                tensor_parallel_size = self.ngpu,
                gpu_memory_utilization = 0.9,
                max_num_batched_tokens = 8192,
                max_model_len = 8192
                )
        else:
            self.llm = LLM(
                model = self.model_path,
                dtype = self.dtype,
                tensor_parallel_size = self.ngpu,
                gpu_memory_utilization = 0.9,
                max_num_batched_tokens = self.context_length,
                )
        
        self.tokenizer = self.llm.get_tokenizer()
    

    def _set_parameters(self, model_config: dict) -> None:
        """Set parameters from the model configuration"""
        
        # default values for parameters if none exists
        defaults = {
            "context_length": 4096,
            "max_new_tokens": 2048,
            "temperature": 0.0,
            "top_p": 1.0,
            "stop": None,
            "do_sample": False
        }
        
        parameters = model_config.get("model_params", {})
        for key, default in defaults.items():
            setattr(self, key, parameters.get(key, default))


    def _set_configs(self, model_config: dict) -> None:
        """Set model hardware configuration."""

        # mapping from string to torch.dtype
        dtype_map = {
            "float32": self.torch.float32,
            "float16": self.torch.float16,
            "bfloat16": self.torch.bfloat16,
            "int8": self.torch.int8,
        }

        # extract inner config if it exists
        configs = model_config.get("model_config", {})

        # get and parse torch_dtype
        d_type = configs.get("dtype", "float32")
        if isinstance(d_type, str):
            if d_type in dtype_map:
                self.dtype = dtype_map[d_type]
            else:
                raise ValueError(f"Unsupported dtype string: '{d_type}'. Must be one of {list(dtype_map.keys())}.")
        elif isinstance(d_type, self.torch.dtype):
            self.dtype = d_type
        else:
            raise TypeError("dtype must be a string or torch.dtype instance.")
        
        ngpu = configs.get("ngpu", 1)
        if isinstance(ngpu, int):
            self.ngpu = ngpu
        else:
            raise TypeError("ngpu must be an integer.")
        

    def generate_prompt(self, system_message, prompt):
        """Generate prompt structure for specific LLM."""
        system_message = system_message or "You are a PDDL coding assistant. Provide concise, correct code only.\n"
        model_name = self.model_engine.lower()

        # try to find matching template key from the prompt_templates
        for key in prompt_templates:
            if key in model_name:
                template = prompt_templates[key]
                return template.format(system_prompt=system_message, prompt=prompt).strip()

        # default fallback if no template matched
        return prompt
    
    @override
    def query(
        self, 
        prompt: str,
        system_prompt: str = None,
        end_when_error: bool=False,
        max_retry: int=3,
        est_margin: int=200,
        ) -> str:
        """Generate a response from model based on the prompt."""
        
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")
        
        full_prompt = self.generate_prompt(system_prompt, prompt)
        assert full_prompt is not None

        input = self.tokenizer(full_prompt)
        requested_tokens = len(input["input_ids"])

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
                print(f"[INFO] connecting to {self.model_engine} ({requested_tokens} tokens)...")
                if requested_tokens >= self.context_length:
                    print(
                        f"[WARNING] Prompt is {requested_tokens} tokens and exceeds context length "
                        f"({self.context_length}). It will be truncated."
                    )
                
                llm_output = self.llm.generate([full_prompt], self.sampling_params)
                llm_output = llm_output[0].outputs[0].text
                    
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
        output_ids = self.tokenizer(llm_output)
        output_tokens = len(output_ids["input_ids"])

        # record token counts
        self.in_tokens += requested_tokens
        self.out_tokens += output_tokens

        self.query_log.append({
            "model": self.model_engine,
            "prompt_tokens": requested_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": requested_tokens + output_tokens,
            "prompt": full_prompt,
            "output": llm_output,
        })
    
        return llm_output
    
    def get_tokens(self) -> tuple[int,int]:
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
            