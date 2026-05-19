"""
PDDL Feedback Generation Functions

This module defines the `FeedbackBuilder` class for constructing structured
feedback loops during PDDL generation using LLMs. It provides default
implementations using prompts from `l2p/templates/feedback/`.
Users can subclass to add custom feedback methods or override hooks.
"""

import json
import time
from typing import Any, Dict, Optional, Tuple, Union, TypeVar, Type

from l2p.llm import BaseLLM, require_llm
from l2p.utils.pddl_format import *
from l2p.utils.pddl_prompt import DEF_FB_PROMPTS, build_ctx, safe_format, jsonify_components
from l2p.utils.pddl_parser import parse_xml_tags, parse_component

T = TypeVar('T', bound=BaseModel)

class FeedbackBuilder:
    """
    Concrete feedback builder for PDDL generation feedback loops.

    Provides default implementations using prompts from ``l2p/templates/feedback/``.
    Subclass to add custom feedback methods or override the three extension hooks:

    * ``resolve_template`` — choose which prompt template to use
    * ``build_prompt`` — fill placeholders into the template
    """

    default_prompts = DEF_FB_PROMPTS

    # ------------------------------------------------------------------
    # Extension hooks (override in subclasses for custom behaviour)
    # ------------------------------------------------------------------

    def resolve_template(self, feedback_type: str, prompt_template: Optional[str] = None) -> str:
        """
        Return the prompt template for *feedback_type*.

        If *prompt_template* is provided it is returned as-is.
        Otherwise the method looks up ``self.default_prompts``.
        """
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
        """
        Inject *placeholders* into *template* and return the result.

        Override this to use a custom templating engine.
        """
        return safe_format(template, **placeholders)

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
        """
        Core feedback loop: resolve template → build prompt → query → parse.

        *output_as_raw* — if ``True`` the raw JSON string inside the XML tag is
        returned instead of a parsed dict (used by :meth:`revise`).
        """
        template = self.resolve_template(feedback_type, prompt_template)
        # Include xml_tag as a prompt placeholder so templates with {xml_tag} (e.g. revise) work
        prompt = self.build_prompt(template, xml_tag=xml_tag, **placeholders)

        print(prompt)

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
                
                blocks = parse_xml_tags(llm_output, xml_tag)
                if not blocks:
                    raise ValueError(
                        f"[ERROR] Missing expected <{xml_tag}> block in LLM output."
                        f"\nLLM Output:\n{llm_output}"
                    )
                result = json.loads(blocks[0])
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
        errors: list[str] | str,
        generated_output: Union[List[Type[T]], Type[T]],
        prompt_template: Optional[str] = None,
        xml_tag: Optional[str] = None,
        description: Optional[str] = None,
        max_retries: int = 3,
        **ctx_kwargs: Any,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Analyse validator errors and produce a structured diagnosis / repair plan.

        Default XML tag: ``<diagnosis>``

        Args:
            model: LLM instance.
            description: Original natural-language description of the domain/problem.
            errors: Validator error messages.
            generated_output: The failed PDDL generation (raw text).
            prompt_template: Override the default prompt template.
            max_retries: Maximum number of retry attempts.
            **ctx_kwargs: Extra context (e.g. ``types=[...]``, ``predicates=[...]``).

        Returns:
            ``(diagnosis_dict, raw_llm_output)``
        """

        # retrieve prompt template (None retrieves default)
        if not prompt_template:
            prompt_template = self.default_prompts.diagnosis
            if not prompt_template:
                raise ValueError(f"[ERROR] No prompt template provided and no default found.")
        
        if not xml_tag:
            xml_tag = "diagnosis"

        print(generated_output)
            
        # inject context in placeholders
        prompt = safe_format(
            template=prompt_template,
            errors=errors,
            generated_output=generated_output,
            xml_tag=xml_tag,
            description=description,
            context=build_ctx(**ctx_kwargs)
        )

        print(prompt)

        # return self._run_feedback(
        #     model=model,

        # )

        # return self._run_feedback(
        #     model=model,
        #     feedback_type="diagnosis",
        #     xml_tag="diagnosis",
        #     prompt_template=prompt_template,
        #     max_retries=max_retries,
        #     output_as_raw=False,
        #     description=description,
        #     context=build_ctx(**ctx_kwargs),
        #     errors=errors,
        #     generated_output=generated_output,
        # )

    @require_llm
    def evaluate(
        self,
        model: BaseLLM,
        description: str,
        generated_output: str,
        prompt_template: Optional[str] = None,
        max_retries: int = 3,
        **ctx_kwargs: Any,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Evaluate a generated PDDL component for semantic quality.

        Default XML tag: ``<evaluation>``

        Returns:
            ``(evaluation_dict, raw_llm_output)``
        """
        return self._run_feedback(
            model=model,
            feedback_type="evaluate",
            xml_tag="evaluation",
            prompt_template=prompt_template,
            max_retries=max_retries,
            output_as_raw=False,
            description=description,
            context=build_ctx(**ctx_kwargs),
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
        **ctx_kwargs: Any,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Extract generalised lessons from a specific failure.

        Default XML tag: ``<reflection>``

        Returns:
            ``(reflection_dict, raw_llm_output)``
        """
        return self._run_feedback(
            model=model,
            feedback_type="reflection",
            xml_tag="reflection",
            prompt_template=prompt_template,
            max_retries=max_retries,
            output_as_raw=False,
            description=description,
            context=build_ctx(**ctx_kwargs),
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
        **ctx_kwargs: Any,
    ) -> Tuple[str, str]:
        """
        Revise a failed PDDL generation by following a repair plan.

        Unlike other methods, the XML tag is dynamic (e.g. ``<types>``) and
        the result is the **raw JSON string** inside the tag rather than a parsed dict.

        Returns:
            ``(corrected_json_string, raw_llm_output)``
        """
        return self._run_feedback(
            model=model,
            feedback_type="revise",
            xml_tag=xml_tag,
            prompt_template=prompt_template,
            max_retries=max_retries,
            output_as_raw=True,
            description=description,
            context=build_ctx(**ctx_kwargs),
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
        **ctx_kwargs: Any,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Select the best candidate from multiple PDDL generations.

        Default XML tag: ``<selection>``

        Returns:
            ``(selection_dict, raw_llm_output)``
        """
        return self._run_feedback(
            model=model,
            feedback_type="select",
            xml_tag="selection",
            prompt_template=prompt_template,
            max_retries=max_retries,
            output_as_raw=False,
            original_prompt=original_prompt,
            context=build_ctx(**ctx_kwargs),
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
        **ctx_kwargs: Any,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Diagnose why a planner failed to find a solution.

        Default XML tag: ``<plan_diagnosis>``

        Returns:
            ``(plan_diagnosis_dict, raw_llm_output)``
        """
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
        **ctx_kwargs: Any,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Evaluate a successful plan for semantic alignment.

        Default XML tag: ``<plan_evaluation>``

        Returns:
            ``(plan_evaluation_dict, raw_llm_output)``
        """
        return self._run_feedback(
            model=model,
            feedback_type="plan_evaluate",
            xml_tag="plan_evaluation",
            prompt_template=prompt_template,
            max_retries=max_retries,
            output_as_raw=False,
            description=description,
            context=build_ctx(**ctx_kwargs),
            domain=domain_pddl,
            problem=problem_pddl,
            plan=plan,
        )
