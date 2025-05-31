"""
This file uses inputted NL descriptions to generate prompts for LLM. The user does not
have to use this class, but it is generally advisable for ease of use.

    - format (str) should be fixed prompt from: /templates
"""


class PromptBuilder:
    def __init__(
        self,
        role: str = None,
        format: str = None,
        examples: list = None,
        task: str = None,
    ):
        self.role = role  # role for LLM to follow (i.e. PDDL predicate constructor)
        self.format = format  # prompting format
        self.examples = (
            examples if examples is not None else []
        )  # n-shot examples for LLM to follow
        self.task = task  # dynamic placeholder given information to LLM

    def set_role(self, role):
        """Sets the role for the LLM to perform task"""
        self.role = role

    def set_format(self, format):
        """Sets the prompting format for LLM to perform task"""
        self.format = format

    def set_examples(self, example):
        """Appends a shot examples for LLM to follow"""
        self.examples.append(example)

    def set_task(self, task):
        """
        Sets a task for the LLM by providing dynamic placeholders to generate and describe domain components.

        The `task` parameter is a structured input that includes various elements to guide the LLM in understanding
        and executing the task. The task may include descriptions, types, actions, and predicates that the LLM
        will process to generate appropriate outputs.

        Here is an example of a dynamic placeholder:
        '''
        ## Domain
        {domain_desc} - A placeholder for the description of the domain, explaining the context and purpose.
        '''

        Args:
            task (str): A structured string or template containing dynamic placeholders to specify the task.
        """
        self.task = task

    def get_role(self):
        """Returns role of the prompt given"""
        return self.role

    def get_format(self):
        """Returns prompting format of the prompt given"""
        return self.format

    def get_examples(self):
        """Returns list of n-examples of the prompt given"""
        return self.examples

    def get_task(self):
        """Returns dynamic placeholder task prompt"""
        return self.task

    def remove_role(self):
        """Removes role prompt"""
        self.role = None

    def remove_format(self):
        """Removes format prompt"""
        self.format = None

    def remove_examples(self, idx):
        """Removes specific index of example list"""
        del self.examples[idx]

    def remove_task(self):
        """Removes dynamic placeholder task prompt"""
        self.task = None

    def generate_prompt(self):
        """Generates the whole prompt in proper format"""
        prompt = ""

        if self.role:
            prompt += f"[ROLE]: {self.role}\n\n"
            prompt += "------------------------------------------------\n"

        if self.format:
            prompt += f"[FORMAT]: {self.format}\n\n"
            prompt += "------------------------------------------------\n"

        if len(self.examples) > 0:
            prompt += f"[EXAMPLE(S)]:\n"
            for i, example in enumerate(self.examples, 1):
                prompt += f"Example {i}:\n{example}\n\n"

            prompt += "------------------------------------------------\n"

        if self.task:
            prompt += f"[TASK]:\nHere is the task to solve:\n{self.task}\n\n"

        return prompt.strip()
