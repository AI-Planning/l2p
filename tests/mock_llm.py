from l2p.llm.base import BaseLLM


class MockLLM(BaseLLM):
    def __init__(self):
        """
        Initialize with a list of responses to simulate the LLM's outputs.
        """
        self.output = ""

    def query(self, prompt: str):
        """
        Simulates the LLM query response.
        """
        return self.output

    def reset_tokens(self):
        """
        Placeholder for resetting tokens; not needed for testing.
        """
        pass
