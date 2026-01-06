from abc import ABC, abstractmethod


class LLM(ABC):
    def __init__(self):
        self._sys_prompt = {"role": "system", "content": "You are a helpful assistant."}
        self.messages = []
    
    @abstractmethod
    def _generate(self, input: str, messages: list[dict[str, str]] = None) -> str:
        pass
    
    def single_turn(self, input: str) -> str:
        return self._generate(input, messages=[self._sys_prompt])
    
    def multi_turn(self, input: str) -> str:
        response = self._generate(input, messages=self.messages)
        self.messages.append({"role": "assistant", "content": response})
        return response
    
    def reset(self):
        self.messages = [self._sys_prompt]