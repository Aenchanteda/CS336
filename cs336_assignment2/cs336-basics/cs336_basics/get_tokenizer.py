import json
import regex as re
from typing import Iterable, Iterator
from collections.abc import Iterable as ABCIterable


class Tokenizer:
    """
    BPE Tokenizer 类，用于编码文本为 token IDs 和解码 token IDs 为文本
    """
    
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        初始化 Tokenizer
        
        Args:
            vocab: dict[int, bytes] - 词汇表，从 token ID 到 bytes 的映射
            merges: list[tuple[bytes, bytes]] - BPE merges 列表
            special_tokens: list[str] | None - 特殊 token 列表（可选）
        """
        # 转换 vocab 格式：dict[int, bytes] -> dict[tuple(bytes), int]
        # 内部使用 tuple(bytes) 作为 key，因为需要匹配字节序列
        self.vocab = {}
        self.inverse_vocab = {}
        
        # 初始化基础字节词汇表（0-255）
        for i in range(256):
            token_tuple = tuple([i])
            self.vocab[token_tuple] = i
            self.inverse_vocab[i] = token_tuple
        
        for token_id, token_bytes in vocab.items:
            token_tuple = tuple(token_bytes)
            if token_tuple not in self.vocab:
                self.vocab[token_tuple] = token_id
                self.inverse_vocab[token_id] = token_tuple
        
        # 处理 special tokens
        if special_tokens is None:
            special_tokens = []
        
        self.special_tokens = special_tokens
        next_id = max(self.inverse_vocab.keys()) + 1 if self.inverse_vocab else 256
        
        # 如果 special tokens 不在 vocab 中，添加到 vocab
        for special_token in special_tokens:
            special_bytes = tuple(special_token.encode('utf-8'))
            if special_bytes not in self.vocab:
                self.vocab[special_bytes] = next_id
                self.inverse_vocab[next_id] = special_bytes
                next_id += 1
        
        # 保存 merges（用于理解 BPE 合并规则）
        self.merges = merges
        
        # 初始化 GPT-2 风格的预分词器
        self.pre_tokenizer = re.compile(
            r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        )
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        从文件加载 vocab 和 merges，创建 Tokenizer 实例
        
        Args:
            vocab_filepath: vocab JSON 文件路径
            merges_filepath: merges 文本文件路径
            special_tokens: 特殊 token 列表（可选）
        
        Returns:
            Tokenizer 实例
        """
        # 加载 vocab（JSON 格式）
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)
        
        # 转换 vocab 格式：dict[str, int] -> dict[int, bytes]
        # vocab_dict 的格式通常是 {token_string: token_id}
        vocab = {}
        for token_str, token_id in vocab_dict.items():
            # token_str 可能是字符串形式的 token，转换为 bytes
            vocab[int(token_id)] = token_str.encode('utf-8')
        
        # 加载 merges（文本格式，每行两个 token）
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and len(line.split()) == 2:
                    #.split()方法将字符串按分隔符分割成列表
                    token1_str, token2_str = line.split()
                    merges.append((token1_str.encode('utf-8'), token2_str.encode('utf-8')))
        
        # 使用 cls 创建并返回新实例
        return cls(vocab, merges, special_tokens)
    
    def _string_to_bytes(self, text: str) -> list[tuple]:
        """
        将文本转换为字节序列（预分词）
        
        Args:
            text: 输入文本字符串
        
        Returns:
            list[tuple]: 字节序列列表，每个元素是一个字节元组
        """
        bytes_seqs = []
        
        # 处理 special tokens：先分割出 special tokens
        if self.special_tokens:
            escaped_tokens = [re.escape(token) for token in self.special_tokens]
            escaped_pattern = r'|'.join(escaped_tokens)
            # 使用 split 分割文本，保留分隔符
            parts = re.split(f'({escaped_pattern})', text)
        else:
            parts = [text]
        
        for part in parts:
            if not part:
                continue
            
            # 检查是否是 special token
            is_special = False
            if self.special_tokens:
                for special_token in self.special_tokens:
                    if part == special_token:
                        special_bytes = tuple(special_token.encode('utf-8'))
                        bytes_seqs.append(special_bytes)
                        is_special = True
                        break
            
            # 如果不是 special token，使用 GPT-2 预分词器处理
            if not is_special:
                for match in self.pre_tokenizer.finditer(part):
                    match_text = match.group(0)
                    byte_seq = tuple(match_text.encode('utf-8'))
                    bytes_seqs.append(byte_seq)
        
        return bytes_seqs
    
    def _encode_byte_seq(self, byte_seq: tuple) -> list[int]:
        """
        编码单个字节序列为 token IDs（使用最长匹配法）
        
        Args:
            byte_seq: 字节序列元组
        
        Returns:
            list[int]: token ID 列表
        """
        encoded = []
        i = 0
        
        while i < len(byte_seq):
            # 尝试匹配最长子词（从最长到最短尝试）
            matched = False
            # 从最长可能的匹配开始（从当前位置到序列末尾）
            for j in range(len(byte_seq), i, -1):
                subword = byte_seq[i:j]
                if subword in self.vocab:
                    encoded.append(self.vocab[subword])
                    i = j
                    matched = True
                    break
            
            # 如果没匹配到，使用单个字节（必然在词汇表中，因为基础字节0-255都在）
            if not matched:
                single_byte = (byte_seq[i],)
                if single_byte in self.vocab:
                    encoded.append(self.vocab[single_byte])
                else:
                    # 如果单个字节也不在（不应该发生），跳过
                    pass
                i += 1
        
        return encoded
    
    def encode(self, text: str) -> list[int]:
        """
        编码文本为 token IDs
        
        Args:
            text: 输入文本字符串
        
        Returns:
            list[int]: token ID 列表
        """
        # 1. 将文本转换为字节序列
        bytes_seqs = self._string_to_bytes(text)
        
        # 2. 编码每个字节序列
        token_ids = []
        for byte_seq in bytes_seqs:
            # 检查是否是 special token
            if byte_seq in self.vocab:
                token_ids.append(self.vocab[byte_seq])
            else:
                # 使用最长匹配法编码
                encoded_seq = self._encode_byte_seq(byte_seq)
                token_ids.extend(encoded_seq)
        
        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        批量编码文本（懒加载，返回生成器）
        
        Args:
            iterable: 文本迭代器（如文件句柄）
        
        Returns:
            Iterator[int]: token ID 的迭代器
        """
        for text in iterable:
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id
    
    def decode(self, ids: list[int]) -> str:
        """
        解码 token IDs 为文本
        
        Args:
            ids: token ID 列表
        
        Returns:
            str: 解码后的文本字符串
        """
        byte_seq = []
        
        # 创建 special tokens 的 bytes 集合，用于快速检查
        special_token_bytes_set = set()
        for special_token in self.special_tokens:
            special_token_bytes_set.add(tuple(special_token.encode('utf-8')))
        
        for token_id in ids:
            if token_id in self.inverse_vocab:
                token_bytes = self.inverse_vocab[token_id]
                # 检查是否是 special token（跳过，不输出）
                if token_bytes not in special_token_bytes_set:
                    byte_seq.extend(token_bytes)
        
        # 将字节序列解码为文本
        try:
            text = bytes(byte_seq).decode('utf-8', errors='replace')
        except Exception:
            text = ""
        
        return text
