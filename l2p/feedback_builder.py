"""
PDDL Feedback Generation Functions

This module defines the `FeedbackBuilder` class for constructing structured
feedback loops during PDDL generation using LLMs. It provides default
implementations using prompts from `l2p/templates/feedback/`.
Users can subclass to add custom feedback methods or override hooks.
"""

import json
import time
from typing import Any, Dict, Optional, Tuple, Type, Union
from l2p.llm import BaseLLM, require_llm
from l2p.utils.pddl_format import *
from l2p.utils.pddl_prompt import DEF_FB_PROMPTS, build_ctx, safe_format
from l2p.utils.pddl_parser import parse_xml_tags


class FeedbackBuilder:

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def normalize_artifacts(self, artifact: Dict[Type[BaseModel], List[BaseModel]] | 
                            List[BaseModel] | BaseModel) -> str:
        """Normalizes the artifact string."""
        if isinstance(artifact, BaseModel): # single model
            data = artifact.model_dump(exclude_none=True)
            
        elif isinstance(artifact, list): # list of models
            data = [item.model_dump(exclude_none=True) for item in artifact]
            
        elif isinstance(artifact, dict): # full dictionary
            data = {
                cls.__name__: [item.model_dump(exclude_none=True) for item in models]
                for cls, models in artifact.items()
            }
        else:
            raise ValueError("Unsupported artifact type provided.")

        return json.dumps(data, indent=4)

    def _run_feedback(
        self,
        model: BaseLLM,
        xml_tag: str,
        prompt: str,
        max_retries: int
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        
        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)

                blocks = parse_xml_tags(llm_output, xml_tag)
                if not blocks:
                    raise ValueError(
                        f"[ERROR] Missing <{xml_tag}> block in LLM output."
                    )
                return blocks[0], llm_output

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output:\n\n{llm_output if 'llm_output' in locals() else 'None'}\n\nRetrying..."
                )
                time.sleep(2)

        raise RuntimeError(
            f"Max retries ({max_retries}) exceeded for '{xml_tag}' feedback."
        )

    # ------------------------------------------------------------------
    # Public feedback methods
    # ------------------------------------------------------------------

    @require_llm
    def llm_diagnose(
        self,
        model: BaseLLM,
        artifact: Dict[Type[BaseModel], List[BaseModel]] | List[BaseModel] | BaseModel,
        errors: List[str] | str,
        xml_tag: Optional[str] = None,
        prompt_template: Optional[str] = None,
        description: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        """
        Runs a diagnostic analysis on the error message. Note: the LLM is simply diagnosing
        the error based on the context of the artifact. Therefore, the LLM should not be
        outputting any PDDL content, but rather, a repair plan to fix it.
        """

        xml_tag_str = xml_tag if xml_tag else "diagnosis"
        errors_str = "\n".join(errors) if isinstance(errors, list) else errors
        artifact_str = self.normalize_artifacts(artifact=artifact)
        prompt_template = prompt_template if prompt_template else DEF_FB_PROMPTS.diagnosis

        prompt = safe_format(
            template=prompt_template,
            xml_tag=xml_tag_str,
            description=description,
            context=build_ctx(**kwargs),
            errors=errors_str,
            artifact=artifact_str
        )

        result, llm_output = self._run_feedback(
            model=model,
            xml_tag=xml_tag_str,
            prompt=prompt,
            max_retries=max_retries
        )

        return result, llm_output
    

    @require_llm
    def llm_evaluate(
        self,
        model: BaseLLM,
        artifact: Dict[Type[BaseModel], List[BaseModel]] | List[BaseModel] | BaseModel,
        xml_tag: Optional[str] = None,
        prompt_template: Optional[str] = None,
        description: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        
        xml_tag_str = xml_tag if xml_tag else "evaluation"
        artifact_str = self.normalize_artifacts(artifact=artifact)
        prompt_template = prompt_template if prompt_template else DEF_FB_PROMPTS.evaluate

        prompt = safe_format(
            template=prompt_template,
            xml_tag=xml_tag_str,
            description=description,
            context=build_ctx(**kwargs),
            artifact=artifact_str
        )

        result, llm_output = self._run_feedback(
            model=model,
            xml_tag=xml_tag_str,
            prompt=prompt,
            max_retries=max_retries
        )

        return result, llm_output
    

    @require_llm
    def llm_reflect(
        self,
        model: BaseLLM,
        artifact: Dict[Type[BaseModel], List[BaseModel]] | List[BaseModel] | BaseModel,
        xml_tag: Optional[str] = None,
        prompt_template: Optional[str] = None,
        diagnosis: Optional[str] = None,
        description: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        
        xml_tag_str = xml_tag if xml_tag else "reflection"
        artifact_str = self.normalize_artifacts(artifact=artifact)
        prompt_template = prompt_template if prompt_template else DEF_FB_PROMPTS.reflection

        prompt = safe_format(
            template=prompt_template,
            xml_tag=xml_tag_str,
            description=description,
            context=build_ctx(**kwargs),
            diagnosis=diagnosis,
            artifact=artifact_str
        )

        result, llm_output = self._run_feedback(
            model=model,
            xml_tag=xml_tag_str,
            prompt=prompt,
            max_retries=max_retries
        )

        return result, llm_output
    

    @require_llm
    def llm_revise(
        self,
        model: BaseLLM,
        artifact: Dict[Type[BaseModel], List[BaseModel]] | List[BaseModel] | BaseModel,
        xml_tag: Optional[str] = None,
        prompt_template: Optional[str] = None,
        diagnosis: Optional[str] = None,
        description: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        
        xml_tag_str = xml_tag if xml_tag else "reflection"
        artifact_str = self.normalize_artifacts(artifact=artifact)
        prompt_template = prompt_template if prompt_template else DEF_FB_PROMPTS.revise

        prompt = safe_format(
            template=prompt_template,
            xml_tag=xml_tag_str,
            description=description,
            context=build_ctx(**kwargs),
            diagnosis=diagnosis,
            artifact=artifact_str
        )

        result, llm_output = self._run_feedback(
            model=model,
            xml_tag=xml_tag_str,
            prompt=prompt,
            max_retries=max_retries
        )

        return result, llm_output
        
        
    @require_llm
    def llm_select(
        self,
        model: BaseLLM,
        candidates: List[BaseModel],
        xml_tag: Optional[str] = None,
        prompt_template: Optional[str] = None,
        description: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        
        xml_tag_str = xml_tag if xml_tag else "selection"
        candidates = "FIX HERE"
        prompt_template = prompt_template if prompt_template else DEF_FB_PROMPTS.select

        prompt = safe_format(
            template=prompt_template,
            xml_tag=xml_tag_str,
            description=description,
            context=build_ctx(**kwargs),
            candidates=candidates
        )

        result, llm_output = self._run_feedback(
            model=model,
            xml_tag=xml_tag_str,
            prompt=prompt,
            max_retries=max_retries
        )

        return result, llm_output

    @require_llm
    def llm_evaluate_plan(
        self,
        model: BaseLLM,
        plan: str,
        domain: str | DomainDetails,
        problem: str | ProblemDetails,
        xml_tag: Optional[str] = None,
        prompt_template: Optional[str] = None,
        description: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        pass

    @require_llm
    def llm_diagnose_plan(
        self,
        model: BaseLLM,
        plan: str,
        domain: str | DomainDetails,
        problem: str | ProblemDetails,
        xml_tag: Optional[str] = None,
        prompt_template: Optional[str] = None,
        description: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        pass