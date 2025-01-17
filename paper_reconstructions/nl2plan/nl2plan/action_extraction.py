"""
Step 3 (Action Extraction) of NL2Plan

This class queries the LLM to construct the actions (in natural language) for given domain in 
Python dictionary format {'name':'description'}.
"""

from l2p import *


class ActionExtraction:
    def __init__(self):
        self.prompt_template = PromptBuilder()
        self.domain_builder = DomainBuilder()
        self.feedback_builder = FeedbackBuilder()

    def action_extraction(
        self,
        model: LLM,
        domain_desc: str,
        action_extraction_prompt: PromptBuilder,
        type_hierarchy: dict[str, str],
        feedback_prompt: str,
    ) -> dict[str, str]:
        """
        Main function of the action extraction construction step.

        Args:
            - model (LLM): LLM to inquire.
            - domain_desc (str): specific domain description to work off.
            - action_extraction_prompt (PromptBuilder): base prompt to extract actions.
            - feedback_prompt (str): feedback template for LLM to correct output.
        Returns:
            - nl_actions (dict[str,str]): dictionary of extracted actions {'name':'desc'}
        """

        nl_actions, response = self.domain_builder.extract_nl_actions(
            model=model,
            domain_desc=domain_desc,
            prompt_template=action_extraction_prompt.generate_prompt(),
            types=type_hierarchy,
        )

        nl_actions, response_fb = self.feedback_builder.nl_action_feedback(
            model=model,
            domain_desc=domain_desc,
            llm_response=response,
            feedback_template=feedback_prompt,
            feedback_type="llm",
            nl_actions=nl_actions,
            type_hierarchy=type_hierarchy,
        )

        return nl_actions
