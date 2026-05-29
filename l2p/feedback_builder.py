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
from l2p.utils.pddl_parser import parse_xml_tags, parse_component
from l2p.domain_builder import DomainBuilder
from l2p.problem_builder import ProblemBuilder
from l2p.planner_builder import PlanningResult


class FeedbackBuilder:

    def normalize_artifacts(
        self,
        artifact: Dict[Type[BaseModel], List[BaseModel]] | List[BaseModel] | BaseModel,
    ) -> str:
        """Normalizes the artifact into clean JSON wrapped in its corresponding XML tags."""
        injected_strings = []

        if isinstance(artifact, BaseModel):
            tag = artifact.__class__.tag
            tag_name = tag[0] if isinstance(tag, (list, tuple)) else tag
            data = artifact.model_dump(exclude_none=True)
            json_str = json.dumps(data, indent=2)
            injected_strings.append(f"<{tag_name}>\n{json_str}\n</{tag_name}>")

        elif isinstance(artifact, list) and len(artifact) > 0:
            grouped = {}
            for item in artifact:
                cls = item.__class__
                if cls not in grouped:
                    grouped[cls] = []
                grouped[cls].append(item.model_dump(exclude_none=True))

            for cls, models in grouped.items():
                tag = cls.tag
                tag_name = tag[0] if isinstance(tag, (list, tuple)) else tag
                json_str = json.dumps(models, indent=2)
                injected_strings.append(f"<{tag_name}>\n{json_str}\n</{tag_name}>")

        elif isinstance(artifact, dict):
            for cls, models in artifact.items():
                tag = cls.tag
                tag_name = tag[0] if isinstance(tag, (list, tuple)) else tag
                data = [item.model_dump(exclude_none=True) for item in models]
                json_str = json.dumps(data, indent=2)
                injected_strings.append(f"<{tag_name}>\n{json_str}\n</{tag_name}>")

        else:
            raise ValueError("Unsupported artifact type provided or empty list.")

        return "\n\n".join(injected_strings)

    def _run_feedback(
        self, model: BaseLLM, xml_tag: str, prompt: str, max_retries: int
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        """Executes the LLM prompt and extracts the XML block."""
        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)

                if blocks := parse_xml_tags(llm_output, xml_tag):
                    return blocks[0], llm_output
                raise ValueError(f"[ERROR] Missing <{xml_tag}> block in LLM output.")

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}.\nLLM Output:\n{llm_output if 'llm_output' in locals() else 'None'}\nRetrying..."
                )
                time.sleep(2)

        raise RuntimeError(
            f"Max retries ({max_retries}) exceeded for '{xml_tag}' feedback."
        )

    def _build_and_run(
        self,
        model: BaseLLM,
        xml_tag: Optional[str],
        default_tag: str,
        prompt_template: Optional[str],
        default_template: str,
        max_retries: int,
        description: Optional[str],
        ctx_kwargs: Dict[str, Any],
        **format_kwargs,
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        """Core abstraction to format standard prompt templates and run the feedback loop."""
        tag = xml_tag or default_tag
        prompt = safe_format(
            template=prompt_template or default_template,
            xml_tag=tag,
            description=description,
            context=build_ctx(**ctx_kwargs),
            **format_kwargs,
        )

        return self._run_feedback(model, tag, prompt, max_retries)

    # ------------------------------------------------------------------
    # Public feedback methods
    # ------------------------------------------------------------------

    @require_llm
    def llm_diagnose(
        self,
        model: BaseLLM,
        artifact: Any,
        errors: Union[List[str], str],
        xml_tag: Optional[str] = None,
        prompt_template: Optional[str] = None,
        description: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        """
        Diagnoses syntax or structural errors in a generated PDDL component.

        The LLM analyzes deterministic Python errors alongside the failed JSON
        artifact to output a root-cause diagnosis and repair plan. It does not
        generate corrected PDDL code.

        Args:
            model (BaseLLM): The LLM engine to use.
            artifact (Any): The failed Pydantic model(s) that caused the error.
            errors (Union[List[str], str]): The traceback or validation errors.
            xml_tag (Optional[str]): The XML tag expected in the LLM response.
            prompt_template (Optional[str]): Custom prompt template string.
            description (Optional[str]): Original NL description of the task.
            max_retries (int): Maximum attempts to query the LLM.
            **kwargs: Additional context variables for the prompt.

        Returns:
            Tuple[Union[Dict[str, Any], str], str]:
                - The parsed dictionary/string from inside the XML tag.
                - The full raw text generated by the LLM.
        """

        return self._build_and_run(
            model=model,
            xml_tag=xml_tag,
            default_tag="diagnosis",
            prompt_template=prompt_template,
            default_template=DEF_FB_PROMPTS.diagnosis,
            max_retries=max_retries,
            description=description,
            ctx_kwargs=kwargs,
            errors="\n".join(errors) if isinstance(errors, list) else errors,
            artifact=self.normalize_artifacts(artifact),
        )

    @require_llm
    def llm_evaluate(
        self,
        model: BaseLLM,
        artifact: Any,
        xml_tag: Optional[str] = None,
        prompt_template: Optional[str] = None,
        description: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        """
        Evaluates a generated PDDL component against natural language instructions.

        Acts as a semantic judge to determine if the generated code fulfills the
        user's original intent, even if the PDDL syntax is technically correct.

        Args:
            model (BaseLLM): The LLM engine to use.
            artifact (Any): The Pydantic model(s) to evaluate.
            xml_tag (Optional[str]): The XML tag expected in the LLM response.
            prompt_template (Optional[str]): Custom prompt template string.
            description (Optional[str]): Original NL description of the task.
            max_retries (int): Maximum attempts to query the LLM.
            **kwargs: Additional context variables for the prompt.

        Returns:
            Tuple[Union[Dict[str, Any], str], str]:
                - The parsed dictionary/string from inside the XML tag.
                - The full raw text generated by the LLM.
        """

        return self._build_and_run(
            model=model,
            xml_tag=xml_tag,
            default_tag="evaluation",
            prompt_template=prompt_template,
            default_template=DEF_FB_PROMPTS.evaluate,
            max_retries=max_retries,
            description=description,
            ctx_kwargs=kwargs,
            artifact=self.normalize_artifacts(artifact),
        )

    @require_llm
    def llm_reflect(
        self,
        model: BaseLLM,
        artifact: Any,
        xml_tag: Optional[str] = None,
        prompt_template: Optional[str] = None,
        diagnosis: Optional[str] = None,
        description: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        """
        Generates generalized lessons learned from a failure.

        Translates a specific diagnostic repair plan into a durable rule that
        can be stored in memory to prevent the LLM from repeating the mistake.

        Args:
            model (BaseLLM): The LLM engine to use.
            artifact (Any): The failed Pydantic model(s).
            xml_tag (Optional[str]): The XML tag expected in the LLM response.
            prompt_template (Optional[str]): Custom prompt template string.
            diagnosis (Optional[str]): The root-cause diagnosis of the failure.
            description (Optional[str]): Original NL description of the task.
            max_retries (int): Maximum attempts to query the LLM.
            **kwargs: Additional context variables for the prompt.

        Returns:
            Tuple[Union[Dict[str, Any], str], str]:
                - The parsed dictionary/string from inside the XML tag.
                - The full raw text generated by the LLM.
        """

        return self._build_and_run(
            model=model,
            xml_tag=xml_tag,
            default_tag="reflection",
            prompt_template=prompt_template,
            default_template=DEF_FB_PROMPTS.reflection,
            max_retries=max_retries,
            description=description,
            ctx_kwargs=kwargs,
            diagnosis=diagnosis,
            artifact=self.normalize_artifacts(artifact),
        )

    @require_llm
    def llm_revise(
        self,
        model: BaseLLM,
        artifact: Any,
        component_class: Union[Type[BaseModel], List[Type[BaseModel]]],
        prompt_template: Optional[str] = None,
        diagnosis: Optional[str] = None,
        description: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> Tuple[Dict[Type[BaseModel], List[BaseModel]], str]:
        """
        Revises a broken PDDL component based on a repair plan and parses the output
        back into explicit Pydantic models.

        Args:
            model (BaseLLM): The LLM engine to use.
            artifact (Any): The failed Pydantic model(s) to fix.
            component_class (Union[Type[BaseModel], List[Type[BaseModel]]]): The expected
                Pydantic classes to extract from the LLM's revised output.
            prompt_template (Optional[str]): Custom prompt template string.
            diagnosis (Optional[str]): The specific repair plan to follow.
            description (Optional[str]): Original NL description of the task.
            max_retries (int): Maximum attempts to query the LLM.
            **kwargs: Additional context variables for the prompt.

        Returns:
            Tuple[Dict[Type[BaseModel], List[BaseModel]], str]:
                - A dictionary mapping each requested component class to its parsed instances.
                - The full raw text generated by the LLM.
        """

        prompt_template = prompt_template if prompt_template else DEF_FB_PROMPTS.revise
        classes = (
            component_class if isinstance(component_class, list) else [component_class]
        )

        prompt = safe_format(
            template=prompt_template,
            description=description,
            context=build_ctx(**kwargs),
            diagnosis=diagnosis,
            artifact=self.normalize_artifacts(artifact),
        )

        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)
                results = {}

                # iterate over each class the LLM was supposed to fix and extract it
                for cls in classes:
                    tags = [cls.tag] if isinstance(cls.tag, str) else cls.tag

                    for t in tags:
                        if raw_blocks := parse_xml_tags(
                            llm_output=llm_output, tag_name=t
                        ):
                            results[cls] = parse_component(
                                raw_blocks=raw_blocks, model_class=cls, tag_name=t
                            )
                            break
                    else:
                        raise ValueError(
                            f"[ERROR] Missing expected XML block in LLM output. Looked for: {cls.tag}"
                        )

                return results, llm_output

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output: \n\n{llm_output if 'llm_output' in locals() else 'None'}\n\n Retrying..."
                )
                time.sleep(2)

        raise RuntimeError(
            "Max retries exceeded. Failed to revise and extract components."
        )

    @require_llm
    def llm_select(
        self,
        model: BaseLLM,
        candidates: List[BaseModel],
        xml_tag: Optional[str] = None,
        prompt_template: Optional[str] = None,
        description: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        """
        Selects the best PDDL candidate from a generated pool.

        The LLM acts as an arbitration agent, comparing multiple options and
        returning the ID of the highest quality output along with reasoning.

        Args:
            model (BaseLLM): The LLM engine to use.
            candidates (List[BaseModel]): A list of alternative generations.
            xml_tag (Optional[str]): The XML tag expected in the LLM response.
            prompt_template (Optional[str]): Custom prompt template string.
            description (Optional[str]): Original NL description of the task.
            max_retries (int): Maximum attempts to query the LLM.
            **kwargs: Additional context variables for the prompt.

        Returns:
            Tuple[Union[Dict[str, Any], str], str]:
                - The parsed dictionary/string from inside the XML tag.
                - The full raw text generated by the LLM.
        """

        formatted_cands = [
            f"<candidate_{i}>\n{json.dumps(c.model_dump(exclude_none=True), indent=2)}\n</candidate_{i}>"
            for i, c in enumerate(candidates, start=1)
        ]
        return self._build_and_run(
            model=model,
            xml_tag=xml_tag,
            default_tag="selection",
            prompt_template=prompt_template,
            default_template=DEF_FB_PROMPTS.select,
            max_retries=max_retries,
            description=description,
            ctx_kwargs=kwargs,
            candidates="\n\n".join(formatted_cands),
        )

    @require_llm
    def llm_evaluate_plan(
        self,
        model: BaseLLM,
        plan: Union[str, PlanningResult],
        domain: Union[str, DomainDetails],
        problem: Union[str, ProblemDetails],
        xml_tag: Optional[str] = None,
        prompt_template: Optional[str] = None,
        description: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        """
        Evaluates a valid plan to ensure it aligns with human semantic intent.

        Even if an external planner finds a mathematically valid plan, it might exploit
        loopholes. This method checks if the plan's behavior makes real-world sense.

        Args:
            model (BaseLLM): The LLM engine to use.
            plan (Union[str, PlanningResult]): The successful plan to evaluate.
            domain (Union[str, DomainDetails]): The domain string or Pydantic model.
            problem (Union[str, ProblemDetails]): The problem string or Pydantic model.
            xml_tag (Optional[str]): The XML tag expected in the LLM response.
            prompt_template (Optional[str]): Custom prompt template string.
            description (Optional[str]): Original NL description of the task.
            max_retries (int): Maximum attempts to query the LLM.
            **kwargs: Additional context variables for the prompt.

        Returns:
            Tuple[Union[Dict[str, Any], str], str]:
                - The parsed dictionary/string from inside the XML tag.
                - The full raw text generated by the LLM.
        """
        return self._build_and_run(
            model=model,
            xml_tag=xml_tag,
            default_tag="plan_evaluation",
            prompt_template=prompt_template,
            default_template=DEF_FB_PROMPTS.plan_evaluate,
            max_retries=max_retries,
            description=description,
            ctx_kwargs=kwargs,
            domain=(
                DomainBuilder.generate_domain(domain)
                if isinstance(domain, DomainDetails)
                else domain
            ),
            problem=(
                ProblemBuilder.generate_problem(problem)
                if isinstance(problem, ProblemDetails)
                else problem
            ),
            plan=(
                format_plan(plan_list=plan.plan)
                if isinstance(plan, PlanningResult)
                else plan
            ),
        )

    @require_llm
    def llm_diagnose_plan(
        self,
        model: BaseLLM,
        domain: Union[str, DomainDetails],
        problem: Union[str, ProblemDetails],
        plan_error: Union[str, PlanningResult],
        xml_tag: Optional[str] = None,
        prompt_template: Optional[str] = None,
        description: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        """
        Diagnoses why an external planner failed to solve a domain/problem pair.

        The LLM analyzes planner tracebacks (e.g., 'Unsolvable', 'Timeout') against
        the PDDL code to determine where the logical bottleneck exists.

        Args:
            model (BaseLLM): The LLM engine to use.
            domain (Union[str, DomainDetails]): The domain string or Pydantic model.
            problem (Union[str, ProblemDetails]): The problem string or Pydantic model.
            plan_error (Union[str, PlanningResult]): The crash trace or failed PlanningResult.
            xml_tag (Optional[str]): The XML tag expected in the LLM response.
            prompt_template (Optional[str]): Custom prompt template string.
            description (Optional[str]): Original NL description of the task.
            max_retries (int): Maximum attempts to query the LLM.
            **kwargs: Additional context variables for the prompt.

        Returns:
            Tuple[Union[Dict[str, Any], str], str]:
                - The parsed dictionary/string from inside the XML tag.
                - The full raw text generated by the LLM.
        """
        return self._build_and_run(
            model=model,
            xml_tag=xml_tag,
            default_tag="plan_diagnosis",
            prompt_template=prompt_template,
            default_template=DEF_FB_PROMPTS.plan_diagnosis,
            max_retries=max_retries,
            description=description,
            ctx_kwargs=kwargs,
            domain=(
                DomainBuilder.generate_domain(domain)
                if isinstance(domain, DomainDetails)
                else domain
            ),
            problem=(
                ProblemBuilder.generate_problem(problem)
                if isinstance(problem, ProblemDetails)
                else problem
            ),
            error=(
                plan_error.error_message
                if isinstance(plan_error, PlanningResult)
                else plan_error
            ),
        )
