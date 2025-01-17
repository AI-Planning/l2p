"""
Step 1 (Type Extraction) of NL2Plan

This class queries the LLM to construct the types for given domain in Python dictionary format.
"""

from l2p import *


class TypeExtraction:
    def __init__(self):
        self.prompt_template = PromptBuilder()
        self.domain_builder = DomainBuilder()
        self.feedback_builder = FeedbackBuilder()

    def type_extraction(
        self,
        model: LLM,
        domain_desc: str,
        type_extraction_prompt: PromptBuilder,
        feedback_prompt: str,
    ) -> dict[str, str]:
        """
        Main function of the type extraction step.

        Args:
            - model (LLM): LLM to inquire.
            - domain_desc (str): specific domain description to work off.
            - type_extraction_prompt (PromptBuilder): base prompt to extract types.
            - feedback_prompt (str): feedback template for LLM to correct output.
        Returns:
            - types (dict[str,str]): type dictionary
        """

        # prompt LLM to extract types
        types, _ = self.domain_builder.extract_type(
            model=model,
            domain_desc=domain_desc,
            prompt_template=type_extraction_prompt.generate_prompt(),
        )

        # feedback mechanism - returns newly modified types (if needed)
        types, _ = self.feedback_builder.type_feedback(
            model=model,
            domain_desc=domain_desc,
            feedback_template=feedback_prompt,
            feedback_type="llm",
            types=types,
            llm_response="",
        )

        return types
