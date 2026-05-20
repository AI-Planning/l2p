"""
PDDL Feedback Generation Functions

This module defines the `FeedbackBuilder` class for constructing structured
feedback loops during PDDL generation using LLMs. It provides default
implementations using prompts from `l2p/templates/feedback/`.
Users can subclass to add custom feedback methods or override hooks.
"""

import json
import time
from typing import Any, Dict, Optional, Tuple, Union

from l2p.llm import BaseLLM, require_llm
from l2p.utils.pddl_prompt import DEF_FB_PROMPTS, build_ctx, safe_format
from l2p.utils.pddl_parser import parse_xml_tags


class FeedbackBuilder:
    """
    Concrete feedback builder for PDDL generation feedback loops.

    Provides default implementations using prompts from ``l2p/templates/feedback/``.
    Subclass to add custom feedback methods or override the three extension hooks:

    * ``resolve_template`` -- choose which prompt template to use
    * ``build_prompt`` -- fill placeholders into the template
    * ``parse_result`` -- turn raw LLM output into a structured dict
    """

    default_prompts = DEF_FB_PROMPTS

    # ------------------------------------------------------------------
    # Extension hooks (override in subclasses for custom behaviour)
    # ------------------------------------------------------------------

    def resolve_template(self, feedback_type: str, prompt_template: Optional[str] = None) -> str:
        if prompt_template:
            return prompt_template
        template = getattr(self.default_prompts, feedback_type, None)
        if not template:
            raise ValueError(
                f"[ERROR] No default template for '{feedback_type}'. "
                f"Available: {[a for a in dir(self.default_prompts) if not a.startswith('_')]}"
            )
        return template

    def build_prompt(self, template: str, **placeholders: Any) -> str:
        return safe_format(template, **placeholders)

    def parse_result(self, llm_output: str, xml_tag: str) -> Union[Dict[str, Any], str]:
        blocks = parse_xml_tags(llm_output, xml_tag)
        if not blocks:
            raise ValueError(
                f"[ERROR] Missing expected <{xml_tag}> block in LLM output."
                f"\nLLM Output:\n{llm_output}"
            )
        return json.loads(blocks[0])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_feedback(
        self,
        model: BaseLLM,
        feedback_type: str,
        xml_tag: str,
        prompt_template: Optional[str],
        max_retries: int,
        output_as_raw: bool,
        **placeholders: Any,
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        template = self.resolve_template(feedback_type, prompt_template)
        prompt = self.build_prompt(template, xml_tag=xml_tag, **placeholders)

        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)

                if output_as_raw:
                    blocks = parse_xml_tags(llm_output, xml_tag)
                    if not blocks:
                        raise ValueError(
                            f"[ERROR] Missing <{xml_tag}> block in LLM output."
                        )
                    return blocks[0], llm_output

                result = self.parse_result(llm_output, xml_tag)
                return result, llm_output

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output:\n\n{llm_output if 'llm_output' in locals() else 'None'}\n\nRetrying..."
                )
                time.sleep(2)

        raise RuntimeError(
            f"Max retries ({max_retries}) exceeded for '{feedback_type}' feedback."
        )

    # ------------------------------------------------------------------
    # Public feedback methods
    # ------------------------------------------------------------------

    @require_llm
    def diagnose(
        self,
        model: BaseLLM,
        description: str,
        errors: str,
        generated_output: str,
        prompt_template: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], str]:
        return self._run_feedback(
            model=model,
            feedback_type="diagnosis",
            xml_tag="diagnosis",
            prompt_template=prompt_template,
            max_retries=max_retries,
            output_as_raw=False,
            description=description,
            context=build_ctx(**kwargs),
            errors=errors,
            generated_output=generated_output,
        )

    @require_llm
    def evaluate(
        self,
        model: BaseLLM,
        description: str,
        generated_output: str,
        prompt_template: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], str]:
        return self._run_feedback(
            model=model,
            feedback_type="evaluate",
            xml_tag="evaluation",
            prompt_template=prompt_template,
            max_retries=max_retries,
            output_as_raw=False,
            description=description,
            context=build_ctx(**kwargs),
            generated_output=generated_output,
        )

    @require_llm
    def reflect(
        self,
        model: BaseLLM,
        description: str,
        diagnosis: str,
        generated_output: str,
        prompt_template: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], str]:
        return self._run_feedback(
            model=model,
            feedback_type="reflection",
            xml_tag="reflection",
            prompt_template=prompt_template,
            max_retries=max_retries,
            output_as_raw=False,
            description=description,
            context=build_ctx(**kwargs),
            diagnosis=diagnosis,
            generated_output=generated_output,
        )

    @require_llm
    def revise(
        self,
        model: BaseLLM,
        description: str,
        repair_plan: str,
        generated_output: str,
        xml_tag: str,
        prompt_template: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> Tuple[str, str]:
        return self._run_feedback(
            model=model,
            feedback_type="revise",
            xml_tag=xml_tag,
            prompt_template=prompt_template,
            max_retries=max_retries,
            output_as_raw=True,
            description=description,
            context=build_ctx(**kwargs),
            repair_plan=repair_plan,
            generated_output=generated_output,
        )

    @require_llm
    def select_best(
        self,
        model: BaseLLM,
        original_prompt: str,
        candidates: str,
        prompt_template: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], str]:
        return self._run_feedback(
            model=model,
            feedback_type="select",
            xml_tag="selection",
            prompt_template=prompt_template,
            max_retries=max_retries,
            output_as_raw=False,
            original_prompt=original_prompt,
            context=build_ctx(**kwargs),
            candidates=candidates,
        )

    @require_llm
    def plan_diagnosis(
        self,
        model: BaseLLM,
        domain_pddl: str,
        problem_pddl: str,
        planner_output: str,
        prompt_template: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], str]:
        return self._run_feedback(
            model=model,
            feedback_type="plan_diagnosis",
            xml_tag="plan_diagnosis",
            prompt_template=prompt_template,
            max_retries=max_retries,
            output_as_raw=False,
            domain=domain_pddl,
            problem=problem_pddl,
            plan=planner_output,
        )

    @require_llm
    def plan_evaluate(
        self,
        model: BaseLLM,
        description: str,
        domain_pddl: str,
        problem_pddl: str,
        plan: str,
        prompt_template: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], str]:
        return self._run_feedback(
            model=model,
            feedback_type="plan_evaluate",
            xml_tag="plan_evaluation",
            prompt_template=prompt_template,
            max_retries=max_retries,
            output_as_raw=False,
            description=description,
            context=build_ctx(**kwargs),
            domain=domain_pddl,
            problem=problem_pddl,
            plan=plan,
        )
