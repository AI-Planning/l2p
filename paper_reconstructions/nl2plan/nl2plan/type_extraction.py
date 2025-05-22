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
        self.syntax_validator = SyntaxValidator()

    def type_extraction(
        self,
        model: BaseLLM,
        domain_desc: str,
        type_extraction_prompt: PromptBuilder,
        feedback_prompt: str,
    ) -> dict[str, str]:
        """
        Main function of the type extraction step.

        Args:
            - model (BaseLLM): LLM to inquire.
            - domain_desc (str): specific domain description to work off.
            - type_extraction_prompt (PromptBuilder): base prompt to extract types.
            - feedback_prompt (str): feedback template for LLM to correct output.
        Returns:
            - types (dict[str,str]): type dictionary
        """
        
        self.syntax_validator.error_types = ['validate_format_types']

        i = 0
        max_feedback_retries = 3
        no_feedback = False
        llm_input_prompt = type_extraction_prompt.generate_prompt()

        while no_feedback == False and i < max_feedback_retries:
            # inner loop: repeat until syntax validator passes
            valid = False
            while not valid:
                types, llm_output, validation_info = self.domain_builder.formalize_types(
                    model=model,
                    domain_desc=domain_desc,
                    prompt_template=llm_input_prompt,
                    syntax_validator=self.syntax_validator
                )
                
                valid = validation_info[0]
                if valid == False:
                    llm_input_prompt = self.generate_validation_prompt(
                        domain_desc=domain_desc,
                        original_llm_output=llm_output,
                        validation_info=validation_info
                    )

                # feedback mechanism: after valid generation
                no_feedback, fb_msg = self.feedback_builder.type_feedback(
                    model=model,
                    domain_desc=domain_desc,
                    llm_output=llm_output,
                    feedback_template=feedback_prompt,
                    feedback_type="llm",
                    types=types
                )
                
                print(no_feedback)
                print(fb_msg)
                
                if no_feedback == False:
                    llm_input_prompt = self.generate_feedback_revision_prompt(
                        fb_msg=fb_msg,
                        types=types
                    )
                    i += 1
                
        return types
    
    
    def generate_validation_prompt(
        self,
        domain_desc: str,
        original_llm_output: str,
        validation_info: tuple[bool, str]
    ) -> str:
        prompt = load_file("paper_reconstructions/nl2plan/prompts/type_extraction/validation.txt")
        prompt_data = {
            "error_msg": validation_info[1],
            "llm_response": original_llm_output,
            "domain_desc": domain_desc,
        }
        prompt = prompt.format(**prompt_data)
        return prompt
    
    
    def generate_feedback_revision_prompt(
        self,
        fb_msg: str,
        types: dict[str,str]
    ) -> str:
        prompt = load_file("paper_reconstructions/nl2plan/prompts/type_extraction/feedback_revision.txt")
        prompt_data = {
            "fb_msg": fb_msg,
            "types": pretty_print_dict(types),
        }
        prompt = prompt.format(**prompt_data)
        
        print("THIS IS THE FEEDBACK REVISION PROMPT:")
        print(prompt)
        
        return prompt