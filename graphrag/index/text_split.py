from abc import ABC, abstractmethod

import tiktoken


class TextSplitter(ABC):
    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        pass


class TokenTextSplitter(TextSplitter):
    def __init__(self, chunk_size: int, over_lap: int, encoding_name: str) -> None:
        self._tokenizer = tiktoken.get_encoding(encoding_name)
        self._chunk_size = chunk_size
        self._over_lap = over_lap
    
    def num_tokens(self, text: str) -> int:
        """Return the number of tokens in a string."""
        return len(self._tokenizer.encode(text))
    
    def split_text(self, text: str) -> list[str]:
        splits = []
        input_ids = self._tokenizer.encode(text=text)
        for i in range(0, len(input_ids), self._chunk_size - self._over_lap):
            chunk_ids = input_ids[i:i+self._chunk_size]
            splits.append(self._tokenizer.decode(chunk_ids))
        return splits