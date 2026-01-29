from ast import Pass
import regex as re
import sys
import os
from collections import defaultdict
from cs336_basics.BPE import input_file as input_path
import os
from typing import BinaryIO
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
#from .BPE import input_file as input_file_path
#如果文件包含if main可能直接被运行，而直接运行脚本时python不会将其视为包的一部分，相对导入只能在包内使用，不能直接用于运行的脚本。


def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)#移动到文件末尾，file.seek(offset, whence=0)，whence=0/os.SEEK_SET表示从文件开头开始，whence=1/os.SEEK_CUR表示从当前位置开始，whence=2/os.SEEK_END 表示从文件末尾开始, offset = 0 表示偏移量
        #.seek（）对象的使用主体，有文件对象（通过open()打开），io模式的文件对象（io.BytesIO()），字符串、列表、普通对象无法打开。
        file_size = file.tell()#返回当前文件指针的位置，由于指针在末尾所以返回值就是文件大小
        file.seek(0)#移动到文件开头，表示从文件开头偏移0字节，等价于file.seel(0,os.SEEK_SET)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size#确保最后一个边界是文件末尾

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk，返回bytes类型，值为读取到的字节数据，如果到达文件末尾则返回空字节b""

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)#返回特殊token在mini_chunk中的位置，找到则返回第一次出现的索引位置为int类型，如果未找到则返回-1
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))#set()函数用于创建一个无序不重复元素集，也就是集合。把列表转换为集合，自动去重，然后转换回列表，并排序。sorted排序从小到大


class train_BPETokenizer:
    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str]):
        self.input_path = input_path
        self.special_tokens = special_tokens
        self.vocab_size = vocab_size
        self.merges = []
        self.pairs_and_counts = defaultdict(int) #不存在的键会被自动创建并初始化为0，通常和+=操作一起。因为如果是普通dict，键盘不存在时会抛KeyError
    


    
    def string_2_bytes(self, text=None):
        """
        将文本转换为字节序列
        Args:
            text: 可选，如果提供文本字符串，则处理文本；如果为None，则从文件读取
        Returns:
            bytes_seqs: 字节序列列表
        """
        all_docs = []
        tokenizer = self.get_gpt2_tokenizer()
        escaped_tokens = [re.escape(token) for token in self.special_tokens]
        escaped_pattern = r'|'.join(escaped_tokens)
        
        if text is not None:
            parts = re.split(escaped_pattern, text)
            # 如果提供了文本，直接处理文本
            escaped_tokens = [re.escape(token) for token in self.special_tokens]
            escaped_pattern = r'|'.join(escaped_tokens)
            parts = re.split(escaped_pattern, text)

            for part in parts:
                if part:
                    doc_token = []
                    for match in tokenizer.finditer(part):
                        match_text = match.group(0)
                        byte_seq = tuple(match_text.encode('utf-8'))
                        doc_token.append(byte_seq)
                    all_docs.append(doc_token)
            return all_docs

        else:
            # 如果没有提供文本，从文件读取
            ## Usage
            input_file = self.input_path
            chunks = []
            with open(input_file, "rb") as f:
                num_processes = 4
                boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
                # The following is a serial implementation, but you can parallelize this
                # by sending each start/end pair to a set of processes.
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    f.seek(start)
                    chunk = f.read(end - start).decode("utf-8", errors="ignore")
                    chunks.append(chunk)
                    # Run pre-tokenization on your chunk and store the counts for each pre-token
            for chunk in chunks:
                if chunk:
                    doc_token = []
                    for match in tokenizer.finditer(chunk):
                        match_text = match.group(0)
                        byte_seq = tuple(match_text.encode('utf-8'))
                        doc_token.append(byte_seq)
                    all_docs.append(doc_token)
            return all_docs
    
    
    def get_gpt2_tokenizer(self):
        
        re_compiled = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        
        return re_compiled

    def init_vocab(self):
        # 基础字节词表
        self.vocab = {tuple([i]): i for i in range(256)}  #采用字典推导式， 词汇表字典：key元组,value数字
        self.inverse_vocab = {i:tuple([i]) for i in range(256)}  #逆向，key数字,value元组
        special_tokens = self.add_special_tokens()
        # 为 special tokens 预留固定 ID（从 256 开始）
        next_id = len(self.vocab)#256
        for tok in special_tokens:
            key = tuple(tok.encode('utf-8'))
            if key not in self.vocab:
                self.vocab[key] = next_id
                self.inverse_vocab[next_id] = key
                next_id += 1
        return self.vocab, self.inverse_vocab
    
    def add_special_tokens(self):
        special_tokens= \
                        ["<|endoftext|>"
     
                        ]
        return special_tokens

    def _flatten_byte_seq(self, byte_seq):
        """将字节序列展平为单个字节列表"""
        flat = []
        if isinstance(byte_seq, int):
            return [byte_seq]
        for item in byte_seq:
            if isinstance(item, tuple) and len(item) > 1:
                flat.extend(item)  # 展平已合并的字节对 
            elif isinstance(item, tuple):
                flat.append(item[0])  # 单元素元组
            else:
                flat.append(item)  # 直接是 int，未合并的字节
        return flat

    def count_bytes_pairs(self, bytes_seqs=None, text=None):
        """
        统计字节对频率
        Args:
            bytes_seqs: 可选，如果提供字节序列列表，则直接使用；如果为None，则从text或文件读取
            text: 可选，如果提供文本字符串，则处理文本；如果为None，则从文件读取
        """
        if bytes_seqs is None:
            bytes_seqs = self.string_2_bytes(text)

        for doc_tokens in bytes_seqs:
            flat_doc = []
            # 展平字节序列以正确统计字节对
            for byte_seq in doc_tokens:
                flat_seq = self._flatten_byte_seq(byte_seq)
                flat_doc.extend(flat_seq)


            for i in range(len(flat_doc)-1):
                pairs = (flat_doc[i], flat_doc[i+1])
                self.pairs_and_counts[pairs] += 1

    def merge_pairs(self, bytes_seqs, best_pair):
        """合并字节对"""
        merged_docs = []
        for doc_tokens in bytes_seqs:
            flat_doc = []
            for byte_seq in doc_tokens:
                flat_seq = self._flatten_byte_seq(byte_seq)
                flat_doc.extend(flat_seq)

            new_doc = []
            i = 0
            while i < len(flat_doc):
                if i < len(flat_doc) - 1 and (flat_doc[i], flat_doc[i+1]) == best_pair:
                    new_doc.append(best_pair)
                    i += 2
                else:
                    new_doc.append(flat_doc[i])
                    i += 1
            merged_docs.append(tuple(new_doc))
        return merged_docs

    
    def bytes_pairs(self, text=None):
        """
        训练BPE：合并字节对
        Args:
            text: 可选，如果提供文本字符串，则处理文本；如果为None，则从文件读取
        """
        bytes_seqs = self.string_2_bytes(text)  # 初始化字节序列
        
        # 调试信息：检查加载的数据量
        total_tokens = sum(len(doc) for doc in bytes_seqs)
        print(f"加载了 {len(bytes_seqs)} 个文档，共 {total_tokens} 个token")
        
        initial_vocab_size = len(self.vocab)
        iteration = 0
        total_needed = self.vocab_size - initial_vocab_size
        
        print(f"初始词汇表大小: {initial_vocab_size}, 目标大小: {self.vocab_size}, 需要合并: {total_needed} 个字节对")
        
        while len(self.vocab) < self.vocab_size:#将词表控制在范围内，不至于merge过大/合无可合
            iteration += 1
            vocab_before = len(self.vocab)
            self.pairs_and_counts.clear()
            self.count_bytes_pairs(bytes_seqs=bytes_seqs)  # 使用已合并的字节序列
            
            if not self.pairs_and_counts:#无更多可合并的byte pair，如果为None，False，0，空字符串、空列表、空字典、空集合等
                print(f"\n无更多可合并的字节对，提前结束。")
                print(f"训练完成！最终词汇表大小: {len(self.vocab)}/{self.vocab_size}")
                break
            
            # 过滤掉已经存在于词汇表中的字节对
            available_pairs = {pair: count for pair, count in self.pairs_and_counts.items() 
                              if pair not in self.vocab}
            
            if not available_pairs:
                print(f"\n所有字节对都已存在于词汇表中，提前结束。")
                print(f"训练完成！最终词汇表大小: {len(self.vocab)}/{self.vocab_size}")
                break
            
            best_pair = max(available_pairs.items(), key=lambda kv:(kv[1],kv[0]))[0]#kv是返回的元组，将key排序规则重新定义为重组的"值、键"。先比第一个元素"值"，降序排列，值相等的话比较"键"（一般str和int升序，越小越高）
            best_count = available_pairs[best_pair]

            pair_bytes = (bytes([best_pair[0]]),bytes([best_pair[1]]))
            self.merges.append(pair_bytes)
            
            #合并
            new_subword = best_pair
            new_subword_id = len(self.vocab)
            self.vocab[new_subword] = new_subword_id
            self.inverse_vocab[new_subword_id] = new_subword
            bytes_seqs = self.merge_pairs(bytes_seqs, best_pair)
            
            vocab_after = len(self.vocab)
            
            # 打印训练进度
            current_progress = vocab_after - initial_vocab_size
            progress_pct = (current_progress / total_needed * 100) if total_needed > 0 else 100
            if iteration % 10 == 0 or iteration == 1:  # 每10次迭代或第一次打印
                print(f"迭代 {iteration:4d} | 词汇表: {vocab_after:5d}/{self.vocab_size:5d} ({progress_pct:.1f}%) | "#f"迭代 {iteration:4d}表示4位宽度的十进制整数
                      f"本次增加: {vocab_after - vocab_before} | 最佳字节对: {best_pair} (频率: {best_count})")
        
        if len(self.vocab) >= self.vocab_size:
            print(f"\n训练完成！词汇表大小已达到目标: {len(self.vocab)}/{self.vocab_size}")

    def _encode_byte_seq(self, byte_seq):
        """
        编码单个字节序列：使用最长匹配法
        从训练好的词汇表中找到最长的匹配子词
        返回字节在字典中匹配结果的token id列表
        """
        encoded = []
        i = 0
        while i < len(byte_seq):
            # 尝试匹配最长子词（从后往前尝试，最多匹配4个字节）
            matched = False
            for j in range(min(i + 4, len(byte_seq)), i, -1):
                subword = tuple(byte_seq[i:j])
                if subword in self.vocab:
                    encoded.append(self.vocab[subword])
                    i = j
                    matched = True
                    break
            
            # 如果没匹配到，使用单个字节（必然在词汇表中）
            if not matched:
                single_byte = tuple([byte_seq[i]])
                encoded.append(self.vocab[single_byte])
                i += 1
        
        return encoded

    def encode(self, text):
        """
        将字节编码为 token IDs
        Args:
            text: 输入文本字符串
        Returns:
            token_ids: token ID 列表
        """
        # 1. 将文本转换为字节序列
        bytes_seqs = self.string_2_bytes(text)
        token_ids = []
        special_tokens = self.add_special_tokens()

        for doc_tokens in bytes_seqs:
            for bytes_seq in doc_tokens:
                bytes_str = bytes(bytes_seq).decode('utf-8', errors = 'replace')
                if bytes_str in self.special_tokens:
                    if bytes_seq in self.vocab:
                        token_ids.append(self.vocab[bytes_seq])
                    continue  # 如果是special token，则跳过后续编码步骤，继续下一个 bytes_seq
                
                # 如果不是special token，使用最长匹配法编码字节序列
                encoded_seq = self._encode_byte_seq(bytes_seq)
                token_ids.extend(encoded_seq)
        
        return token_ids
    
    def encode_text(self,text):  # 调试用
        # bytes 到 str：便于查看
        byte_seqs = self.string_2_bytes(text)
        readable_tokens = []
        special_tokens = self.add_special_tokens()
        for byte_seq in byte_seqs:
            # 特殊 token 直接跳过，保留原始形式
            if byte_seq in special_tokens and self.vocab:
                readable_tokens.append(byte_seq)
                continue#不终止循环，直接进入下一轮循环

            # 确保是可迭代的字节序列
            if not isinstance(byte_seq, tuple):
                byte_seq = (byte_seq,)

            # 将字节元组恢复为字符串，无法解码的字节用 replacement 字符显示
            decoded = bytes(byte_seq).decode("utf-8", errors="replace")
            readable_tokens.append(decoded)

        return readable_tokens
    
    def decode(self, token_ids):
        """
        将 token IDs 解码为文本
        Args:
            token_ids: token ID 列表
        Returns:
            text: 解码后的文本字符串
        """
        if not hasattr(self, 'inverse_vocab'):#检查是否有属性inverse_vocab，如果没有则抛出错误
            raise ValueError("词汇表未初始化，请先调用 train() 或 init_vocab()")
        
        byte_seq = []
        special_tokens = self.add_special_tokens()
        
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                subword = self.inverse_vocab[token_id]
                # 检查是否是特殊 token
                try:
                    subword_str = bytes(subword).decode('utf-8', errors='replace') 
                    if subword_str in special_tokens:
                        continue
                except:
                    Pass
                byte_seq.extend(subword)
            else:
                continue
        try:
            text = bytes(byte_seq).decode('utf-8',errors='replace')
        except:
            text = ""
        return text
    
    def train(self, text=None, save_dir = None):
        """
        训练 Byte-level BPE：从文本生成词汇表
        Args:
            text: 可选，如果提供文本字符串，则使用文本训练；如果为None，则从文件读取
        """
        print("开始训练 BPE...")
        self.get_gpt2_tokenizer()
        self.add_special_tokens()
        self.init_vocab()
        print(f"初始化词汇表完成，大小: {len(self.vocab)}")
        print("开始合并字节对...")
        self.bytes_pairs(text)  # 传递文本参数

        print(f"训练完成！词汇表大小：{len(self.vocab)}")
        vocab_formatted = self.get_vocab()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            vocab_file_path = os.path.join(save_dir, 'vocab.json')
            merges_file_path = os.path.join(save_dir, 'merges.txt')
            print(f"保存 vocab 和 merges 到: {save_dir}")
            self.save(vocab_file_path, merges_file_path)
        return vocab_formatted, self.merges

    def get_vocab(self):
        vocab_formatted = {}
        for key, value in self.vocab.items():
            vocab_formatted[value] = bytes(key)
        return vocab_formatted

    def save(self, vocab_file_path: str, merges_file_path:str):
        """
        vocab_file_path:    保存JSON文件路径
        merges_file_path:   保存文本文件路径
        
        """
        import json
        vocab_json = {}

        for token_id, token_bytes in self.get_vocab().items():
            try:
                token_str = token_bytes.decode('utf-8')
            except UnicodeDecodeError:
                token_str = ''.join(f'\\x{b:02x}' for b in token_bytes)
            vocab_json[str(token_id)] = token_str
        
        with open (vocab_file_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_json, f, ensure_ascii=False, indent=2)
            #json.dump(要序列化的python对象obj, 文件对象fp, 
            # 将非ascii字符转义？ensure_ascii=False保留原字符, 格式化锁紧的空格数indent)
        with open(merges_file_path, 'w', encoding='utf-8') as f:
            for merge_pair in self.merges:
                token1, token2  = merge_pair
                try:
                    token1_str = token1.decode('utf-8')
                    token2_str = token2.decode('utf-8')
                    f.write(f"{token1_str} {token2_str}\n")
                except UnicodeDecodeError:
                    token1_str = ''.join(f'\\x{b:02x}' for b in token1)
                    token2_str = ''.join(f'\\x{b:02x}' for b in token2)
                    f.write(f"{token1_str} {token2_str}\n")
                    #f.write：f必须是文件类型对象，write的内容必须是字符串
    @classmethod
    def load(cls, vocab_file_path, merges_file_path, special_tokens, input_path):
        import json
        instance = cls(input_path=input_path, vocab_size = 0, special_tokens=special_tokens)
        instance.init_vocab()
        with open(vocab_file_path,'r', encoding='utf-8') as f:
            vocab_json = json.load(f)

        for token_id_str, token_str in vocab_json.items():
            token_id = int(token_id_str)
            # 处理转义序列（如 \x00）或普通字符串
            if '\\x' in token_str:
                # 解析转义序列：\x00\xff -> bytes
                import re
                hex_bytes = re.findall(r'\\x([0-9a-fA-F]{2})', token_str)
                token_bytes = bytes(int(h, 16) for h in hex_bytes)
            else:
                token_bytes = token_str.encode('utf-8')
            
            token_tuple = tuple(token_bytes)
            instance.vocab[token_tuple] = token_id
            instance.inverse_vocab[token_id] = token_tuple
        
        # 更新 vocab_size
        instance.vocab_size = len(instance.vocab)
        
        # 加载 merges
        instance.merges = []
        with open(merges_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and len(line.split()) == 2:
                    token1_str, token2_str = line.split()
                    # 处理转义序列
                    if '\\x' in token1_str:
                        hex_bytes = re.findall(r'\\x([0-9a-fA-F]{2})', token1_str)
                        token1 = bytes(int(h, 16) for h in hex_bytes)
                    else:
                        token1 = token1_str.encode('utf-8')
                    
                    if '\\x' in token2_str:
                        hex_bytes = re.findall(r'\\x([0-9a-fA-F]{2})', token2_str)
                        token2 = bytes(int(h, 16) for h in hex_bytes)
                    else:
                        token2 = token2_str.encode('utf-8')
                    
                    instance.merges.append((token1, token2))
        
        return instance








if __name__ == "__main__":
    import os
    
    # 构建正确的文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, '..', 'data', 'Raw', 'TinyStoriesV2-GPT4-valid.txt')
    data_file = os.path.normpath(data_file)  # 规范化路径
    #os.path.normpath(path)：参数：路径字符串（相对/绝对）；返回值：规范化后的路径字符串
    #作用：规范化处理路径，例如多余的斜杠、..、.、平台特定的路径分隔符等等，需要解析为实际路径
    save_dir = os.path.join(current_dir, '..', 'data', 'Processed')
    # 创建并训练 BPE Tokenizer
    bpe = train_BPETokenizer(
        input_path=data_file,
        vocab_size=500,
        special_tokens=["<|endoftext|>"]  # 特殊 token 列表
    )
    bpe.train(save_dir=save_dir)

    test_text = "I love apple and grape!"
    encoded = bpe.encode(test_text)
    print(f"\n测试文本：{test_text}")
    print(f"编码结果（前 10 个 ID）：{encoded[:10]}")
    
    # 解码验证
    decoded = bpe.decode(encoded)
    print(f"解码结果：{decoded}")
    
     # 从文件加载（示例）
    '''print("\n" + "="*50)
    print("从文件加载 tokenizer...")
    loaded_bpe = train_BPETokenizer.load(
        vocab_filepath=os.path.join(save_dir, 'vocab.json'),
        merges_filepath=os.path.join(save_dir, 'merges.txt'),
        special_tokens=["<|endoftext|>"]
    )
    
    # 测试加载的 tokenizer
    encoded2 = loaded_bpe.encode(test_text)
    print(f"加载的 tokenizer 编码结果: {encoded2[:10]}")
    assert encoded == encoded2, "编码结果应该一致！"
    print("✓ 加载成功，编码结果一致！")'''
