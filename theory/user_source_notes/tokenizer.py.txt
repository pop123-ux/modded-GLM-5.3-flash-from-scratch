"""A deterministic byte tokenizer trained w/ no external vocabulary."""
class ByteTokenizer:
    pad_id = 0 # Padding token
    bos_id = 1 # Beginning of Sequence token
    eos_id = 2 # End of Sequence token
    sep_id = 3 # Separator token
    byte_offset = 4 # The shift factor, because IDs 0 through 3 are reserved for special tokens, the actual text bytes must start at ID 4
    vocab_size = 260
    
    """
    A note on appending special tokens:
    
    - Appending special tokens to the end of the vocabulary is a valid alternative, but we choose to append them at the beginning instead because:
    
    1. If special tokens are at the beginning, their IDs never change, no matter how much we alter the core vocab size
    2. When writing neural network code, we frequently need to check for padding or masking, having the padding token at 0 simplifies mathematical operations
    
    """
    
    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        ids = [self.byte_offset + value for value in text.encode('utf-8')] # text.encode('utf-8') converts the string into raw bytes, in UTF-8, characters are represented by 1 to 4 bytes, where each byte is an integer from 0 to 255 | self.byte_offset + value shifts every byte value up to 4, meaning that a byte value of 0 becomes ID 4 and a byte value of 255, becomes ID 259 (completing the vocab size)
        """If bos and eos flags are set to True the code prepends the bos_id to the front of the list or appends the eos_id to the end"""
        if bos:
            ids.insert(0, self.bos_id) # Insert bos token at the beginning of the sequence
        if eos:
            ids.append(self.eos_id) # Insert eos token at the end of the sequence
        return ids
    
    """Codified both texts separatelly (without EOS/BOS)"""
    def encode_pair(self, context: str, question: str) -> list[int]:
        context_ids = [self.byte_offset + v for v in context.encode('utf-8')]
        question_ids = [self.byte_offset + v for v in question.encode('utf-8')]
        
        return [self.bos_id] + context_ids + [self.sep_id] + question_ids + [self.eos_id]

    def decode(self, ids: list[int]) -> str:
        values = bytearray()
        for token in ids:
            if token == self.eos_id: # If it hits an eos_id it breaks immediately
                break
            if token >= self.byte_offset: # If a token ID is 4 or >= self.byte_offset, it subtracts 4 to get the original raw byte value and appends it to the bytearray, safely skipping control tokens like bos_id, pad_id, and sep_id
                values.append(token - self.byte_offset)
        return values.decode('utf-8', errors='replace') # Decodes the tokens back to strings