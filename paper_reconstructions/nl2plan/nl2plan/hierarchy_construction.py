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
        self.syntax_validator = SyntaxValidator()
        self.syntax_validator.error_types = [
            "validate_format_types",
            "validate_cyclic_types",
        ]

    def hierarchy_construction(
        self,
        model: BaseLLM,
        domain_desc: str,
        type_hierarchy_prompt: PromptBuilder,
        types: dict[str, str],
        feedback_prompt: str,
        max_feedback_retries: int = 1,
        max_syntax_retries: int = 3,
    ) -> tuple[list[dict[str, str]], str]:
        """
        Main function of the hierarchy construction step.

        Args:
            - model (BaseLLM): LLM to inquire.
            - domain_desc (str): specific domain description to work off.
            - type_hierarchy_prompt (PromptBuilder): base prompt to organize types.
            - types (dict[str,str]): type dictionary
            - feedback_prompt (str): feedback template for LLM to correct output.
        Returns:
            - type_hierarchy (list[dict[str,str]]): organized hierarchy type dictionary
        """

        i = 0
        no_feedback = False
        llm_input_prompt = type_hierarchy_prompt.generate_prompt()

        # store last valid results
        last_valid_types = None
        last_valid_output = None

        while not no_feedback and i <= max_feedback_retries:
            # inner loop: repeate until syntax validator passes
            max_validation_retries = max_syntax_retries
            valid = False
            while not valid and max_validation_retries > 0:
                type_hierarchy, llm_output, validation_info = (
                    self.domain_builder.formalize_type_hierarchy(
                        model=model,
                        domain_desc=domain_desc,
                        prompt_template=llm_input_prompt,
                        types=types,
                        syntax_validator=self.syntax_validator,
                    )
                )

                valid = validation_info[0]
                if valid:
                    # store last valid results
                    last_valid_types = type_hierarchy
                    last_valid_output = llm_output
                else:
                    llm_input_prompt = self.generate_validation_prompt()
                    max_validation_retries -= 1

            if i < max_feedback_retries:
                # feedback mechanism: after valid generation
                no_feedback, fb_msg = self.feedback_builder.type_feedback(
                    model=model,
                    domain_desc=domain_desc,
                    llm_output=llm_output,
                    feedback_template=feedback_prompt,
                    feedback_type="llm",
                    types=type_hierarchy,
                )
                if not no_feedback:
                    llm_input_prompt = self.generate_feedback_revision_prompt(
                        fb_msg=fb_msg, types=type_hierarchy
                    )
                    i += 1
            else:
                break

        return last_valid_types, last_valid_output

    def generate_validation_prompt(
        self,
        domain_desc: str,
        original_llm_output: str,
        validation_info: tuple[bool, str],
    ) -> str:
        prompt = load_file("paper_reconstructions/nl2plan/prompts/validation.txt")
        prompt = (
            prompt.replace("{error_msg}", validation_info[1])
            .replace("{llm_response}", original_llm_output)
            .replace("{domain_desc}", domain_desc)
        )

        return prompt

    def generate_feedback_revision_prompt(
        self, fb_msg: str, types: dict[str, str]
    ) -> str:
        prompt = load_file(
            "paper_reconstructions/nl2plan/prompts/hierarchy_construction/feedback_revision.txt"
        )
        prompt = prompt.replace("{fb_msg}", fb_msg).replace(
            "{types}", pretty_print_dict(types)
        )

        return prompt
