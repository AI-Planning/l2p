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
        model: BaseLLM,
        domain_desc: str,
        action_extraction_prompt: PromptBuilder,
        type_hierarchy: list[dict[str, str]],
        feedback_prompt: str,
        max_feedback_retries: int = 1,
    ) -> tuple[dict[str, str], str]:
        """
        Main function of the action extraction construction step.

        Args:
            - model (BaseLLM): LLM to inquire.
            - domain_desc (str): specific domain description to work off.
            - action_extraction_prompt (PromptBuilder): base prompt to extract actions.
            - feedback_prompt (str): feedback template for LLM to correct output.
        Returns:
            - nl_actions (dict[str,str]): dictionary of extracted actions {'name':'desc'}
        """

        i = 0
        no_feedback = False
        llm_input_prompt = action_extraction_prompt.generate_prompt()

        while not no_feedback and i <= max_feedback_retries:
            nl_actions, llm_output = self.domain_builder.extract_nl_actions(
                model=model,
                domain_desc=domain_desc,
                prompt_template=llm_input_prompt,
                types=type_hierarchy,
            )

            if i < max_feedback_retries:
                # feedback mechanism
                no_feedback, fb_msg = self.feedback_builder.nl_action_feedback(
                    model=model,
                    domain_desc=domain_desc,
                    llm_output=llm_output,
                    feedback_template=feedback_prompt,
                    feedback_type="llm",
                    types=type_hierarchy,
                    nl_actions=nl_actions,
                )
                if not no_feedback:
                    llm_input_prompt = self.generate_feedback_revision_prompt(
                        fb_msg=fb_msg, nl_actions=nl_actions
                    )
                    i += 1
            else:
                break

        return nl_actions, llm_output

    def generate_feedback_revision_prompt(
        self, fb_msg: str, nl_actions: dict[str, str]
    ) -> str:
        prompt = load_file(
            "paper_reconstructions/nl2plan/prompts/action_extraction/feedback_revision.txt"
        )
        prompt = prompt.replace("{fb_msg}", fb_msg).replace(
            "{nl_actions}", pretty_print_dict(nl_actions)
        )

        return prompt
