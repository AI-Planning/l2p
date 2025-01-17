"""
Step 2 (Hierarchy Construction) of NL2Plan

This class queries the LLM to organize the types for given domain in a nested Python dictionary format.
"""

from l2p import *


class HierarchyConstruction:
    def __init__(self):
        self.prompt_template = PromptBuilder()
        self.domain_builder = DomainBuilder()
        self.feedback_builder = FeedbackBuilder()

    def hierarchy_construction(
        self,
        model: LLM,
        domain_desc: str,
        type_hierarchy_prompt: PromptBuilder,
        types: dict[str, str],
        feedback_prompt: str,
    ) -> dict[str, str]:
        """
        Main function of the hierarchy construction step.

        Args:
            - model (LLM): LLM to inquire.
            - domain_desc (str): specific domain description to work off.
            - type_hierarchy_prompt (PromptBuilder): base prompt to organize types.
            - feedback_prompt (str): feedback template for LLM to correct output.
        Returns:
            - type_hierarchy (dict[str,str]): organized hierarchy type dictionary
        """

        # prompt LLM to extract type hierarchy (using L2P)
        type_hierarchy, _ = self.domain_builder.extract_type_hierarchy(
            model=model,
            domain_desc=domain_desc,
            prompt_template=type_hierarchy_prompt.generate_prompt(),
            types=types,
        )

        # feedback mechanism
        type_hierarchy, _ = self.feedback_builder.type_hierarchy_feedback(
            model=model,
            domain_desc=domain_desc,
            feedback_template=feedback_prompt,
            feedback_type="llm",
            type_hierarchy=type_hierarchy,
            llm_response="",
        )

        return type_hierarchy
