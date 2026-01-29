import regex as re
from collections import defaultdict
import regex as re
import BPE
#from BPE_token.BPE import download_dataset_tinystories_valid

bytes_seqs = self.string_2_bytes()

class BPETokenizer:
    def __init__(self,num_merges=None,vocab_size=30000,vocab=1):
        self.num_merges = num_merges
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.pairs_and_counts = defaultdict(int) #不存在的键会被自动创建并初始化为0，通常和+=操作一起。因为如果是普通dict，键盘不存在时会抛KeyError

    def string_2_bytes(self):
        bytes_seqs = []
        tokenizer = self.get_gpt2_tokenizer()
        input_file = BPE.input_file()
        with open(input_file, 'r',encoding='utf-8') as texts:
            for text in texts:
                for match in tokenizer.finditer(text):
                    match = match.group(0)
                    matches = tuple(match.encode('utf-8'))
                    bytes_seqs.append(matches)
        return
        
    def get_gpt2_tokenizer(self):
        
        re_compiled = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        
        return re_compiled
    

    def init_vocab(self):
        # 基础字节词表
        self.vocab = {tuple([i]): i for i in range(256)}  # 词汇表字典：key元组,value数字
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

    def count_bytes_pairs(self):
        bytes_seqs
        for byte_seq in bytes_seqs:
            for i in range(len(byte_seq)-1):
                pairs = (byte_seq[i],byte_seq[i+1])
                self.pairs_and_counts[pairs] += 1

    def merge_pairs(self,best_pair):
        bytes_seqs = self.string_2_bytes()
        new_merged = []
        for byte in bytes_seqs:
            new_seq = []
            i = 0
            while i < len(bytes_seqs):
                if i < len(bytes_seqs)-1 and (byte[i],byte[i+1] == best_pair):
                    new_seq.append(best_pair)
                    i+=2
                else:
                    new_seq.append(byte[i])
                    i+=1
        new_merged.append(tuple(new_seq))
        return new_merged

    
    def bytes_pairs(self):
        bytes_seqs = self.string_2_bytes()
        while len(self.vocab) < self.vocab.size:#将词表控制在范围内，不至于merge过大/合无可合
            self.pairs_and_counts.clear()
            if not self.count_bytes_pairs():#无更多可合并的byte pair，如果为None，False，0，空字符串、空列表、空字典、空集合等
                break
            best_pair = max(self.pairs_and_counts.items, key=lambda kv:(kv[1],kv[0]))#kv是返回的元组，将key排序规则重新定义为重组的“值、键”。先比第一个元素“值”，降序排列，值相等的话比较“键”（一般str和int升序，越小越高）
            #合并
            new_subword = best_pair
            new_subword_id = len(self.vocab)
            self.vocab[new_subword] = new_subword_id
            self.inverse_vocab[new_subword_id] = new_subword
            bytes_seqs = self.merge_pairs(bytes_seqs,best_pair)

    def encode(self,bytes_seq):
        #字节 到vocab token ID
        bytes_seqs = self.string_2_bytes()
        token_ids = []
        special_tokens = self.add_special_tokens()

        for bytes_seq in bytes_seqs:
            if bytes_seq in special_tokens:
                if bytes_seq in self.vocab:
                    token_ids.append(self.vocab[bytes_seq])
                continue #如果是special token，则直接获取ID

            #如果不是special token,将字节开始merge
            subwords = self.merge_pairs()
            for subword in subwords:
                if subword in self.vocab:
                    token_ids.append(self.vocab[subword])
                else:#如果subword不在词汇表中，尝试char级编码
                    for char in subword:
                        if char in self.vocab:
                            token_ids.append(self.vocab[char])
        return token_ids
    
    def encode_text(self):  # 调试用
        # bytes 到 str：便于查看
        byte_seqs = self.string_2_bytes()
        readable_tokens = []
        special_tokens = self.add_special_tokens()
        for byte_seq in byte_seqs:
            # 特殊 token 直接跳过，保留原始形式
            if self.special_tokens and byte_seq in special_tokens:
                readable_tokens.append(byte_seq)
                continue

            # 确保是可迭代的字节序列
            if not isinstance(byte_seq, tuple):
                byte_seq = (byte_seq,)

            # 将字节元组恢复为字符串，无法解码的字节用 replacement 字符显示
            decoded = bytes(byte_seq).decode("utf-8", errors="replace")
            readable_tokens.append(decoded)

        return readable_tokens
    
    def decode(self):
        # token ID --> 文本
        texts = []
        token_ids = self.encode()
        
        for token_id in token_ids:
            subword = self.inverse_vocab[token_id]
            texts.extend(subword)
        
        #去除special tokens，解码文本
        current_byte = []
        final = []
        special_tokens = self.add_special_tokens()

        for special in texts:
            if special == special_tokens:
                if current_byte:#当遇见特殊值时，如果有内容，就转换并解码成字符串加入final，随后清空为下一轮做准备
                    final.append(bytes(current_byte).decode('utf-8',errors = 'replace'))
                    current_byte = []
            else:
                current_byte.append(special)#非特殊值直接追加，收集一段连续的字节
        return ''.join(final)
    
    def train(self, text):
        """训练 Byte-level BPE：从文本生成词汇表"""
        re_compiled = self.get_gpt2_tokenizer()
        self.string_2_bytes()
        self.add_special_tokens()
        self.init_vocab()
        self.count_bytes_pairs()
        self.bytes_pairs()


        while len(self.vocab) < self.vocab_size:
            # 统计字节对频率
            self.count_bytes_pairs()
            if not self.bytes_pairs:
                break  # 无更多可合并的字节对
            # 找到最高频字节对
            best_pair = max(self.pairs_and_counts.items, key=lambda kv:(kv[1],kv[0]))#kv是返回的元组，将key排序规则重新定义为重组的“值、键”。先比第一个元素“值”，降序排列，值相等的话比较“键”（一般str和int升序，越小越高）
            # 合并字节对，生成新子词，加入词汇表
            new_subword = best_pair
            new_id = len(self.vocab)
            self.vocab[new_subword] = new_id
            self.inverse_vocab[new_id] = new_subword
            # 更新字节序列（用新子词替换原字节对）
            byte_seqs = self._merge_byte_pair(byte_seqs, best_pair)
        print(f"训练完成！词汇表大小：{len(self.vocab)}")        

    
if __name__ == "__main__":
    train_text = """apple banana apple pear orange grape
                    apple is red banana is yellow pear is green
                    I love eating apple and banana"""
    bpe = BPETokenizer(vocab_size=300)
    bpe.train(train_text)

    test_text = "I love apple and grape!"
    encoded = bpe.encode(test_text)
    print(f"\n测试文本：{test_text}")
    print(f"编码结果（前 10 个 ID）：{encoded[:10]}")
    
    # 解码验证
    decoded = bpe.decode(encoded)
    print(f"解码结果：{decoded}")
