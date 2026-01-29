import re
from collections import defaultdict

class ByteLevelBPE:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size  # 目标词汇表大小（初始 256 字节 + 合并生成的子词）
        self.vocab = {}  # 最终词汇表：key=子词（字节元组），value=编码ID
        self.inv_vocab = {}  # 反向映射：编码ID → 子词（字节元组）
        self.byte_pairs = defaultdict(int)  # 字节对频率统计,所有新键第一次被访问时，默认值都是 0

    def _text_to_bytes(self, text):
        """文本 → UTF-8 字节序列（每个词末尾添加终止符 </w>，用字节 255 表示）"""
        words = re.split(r'\s+', text.strip())  # 先用.strip()删除字符串首尾的所有空白字符，然后按照正则分割字符串
        byte_seqs = []
        for word in words:
            # 词 → UTF-8 字节 + 终止符（255）
            byte_seq = list(word.encode('utf-8')) + [255]
            byte_seqs.append(tuple(byte_seq))  # 转元组方便后续处理
        return byte_seqs

    def _init_vocab(self):
        """初始化词汇表：0-255 所有字节"""
        self.vocab = {tuple([i]): i for i in range(256)}
        self.inv_vocab = {i: tuple([i]) for i in range(256)}

    def _count_byte_pairs(self, byte_seqs):
        """统计所有相邻字节对的频率"""
        self.byte_pairs.clear()
        for seq in byte_seqs:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i+1])
                self.byte_pairs[pair] += 1

    def _merge_byte_pair(self, byte_seqs, best_pair):
        """合并最高频字节对，生成新序列"""
        new_byte_seqs = []
        for seq in byte_seqs:
            new_seq = []
            i = 0
            while i < len(seq):
                # 匹配到最佳字节对，合并为新子词（元组形式）
                if i < len(seq) - 1 and (seq[i], seq[i+1]) == best_pair:
                    new_seq.append(best_pair)
                    i += 2
                else:
                    new_seq.append(tuple([seq[i]]))  # 单个字节保持元组形式
                    i += 1
            new_byte_seqs.append(tuple(new_seq))
        return new_byte_seqs

    def train(self, text):
        """训练 Byte-level BPE：从文本生成词汇表"""
        # 1. 文本转字节序列（带终止符）
        byte_seqs = self._text_to_bytes(text)
        # 2. 初始化词汇表（0-255 字节）
        self._init_vocab()
        # 3. 迭代合并高频字节对
        while len(self.vocab) < self.vocab_size:
            # 统计字节对频率
            self._count_byte_pairs(byte_seqs)
            if not self.byte_pairs:
                break  # 无更多可合并的字节对
            # 找到最高频字节对
            best_pair = max(self.byte_pairs, key=self.byte_pairs.get)
            # 合并字节对，生成新子词，加入词汇表
            new_subword = best_pair
            new_id = len(self.vocab)
            self.vocab[new_subword] = new_id
            self.inv_vocab[new_id] = new_subword
            # 更新字节序列（用新子词替换原字节对）
            byte_seqs = self._merge_byte_pair(byte_seqs, best_pair)
        print(f"训练完成！词汇表大小：{len(self.vocab)}")

    def _encode_seq(self, byte_seq):
        """单个字节序列（词）编码：最长匹配法"""
        encoded = []
        i = 0
        while i < len(byte_seq):
            # 尝试匹配最长子词（从当前位置向后最多取 4 个字节，可调整）
            for j in range(min(i + 4, len(byte_seq)), i, -1):
                subword = tuple(byte_seq[i:j])
                if subword in self.vocab:
                    encoded.append(self.vocab[subword])
                    i = j
                    break
            else:
                # 未匹配到任何子词，回退到单个字节（必然存在于词汇表）
                encoded.append(self.vocab[tuple([byte_seq[i]])])
                i += 1
        return encoded

    def encode(self, text):
        """文本编码：将文本转为 Byte-level BPE 编码 ID 序列"""
        byte_seqs = self._text_to_bytes(text)  # 文本转字节序列（带终止符）
        encoded = []
        for seq in byte_seqs:
            # 展平字节序列（原 seq 是元组，转列表方便处理）
            flat_seq = [b for sub in seq for b in sub]
            encoded.extend(self._encode_seq(flat_seq))
        return encoded

    def decode(self, encoded_ids):
        """解码：将编码 ID 序列转回文本"""
        byte_seq = []
        for idx in encoded_ids:
            subword = self.inv_vocab[idx]
            byte_seq.extend(subword)
        # 按终止符 255 分割词，去除终止符后解码为文本
        text_parts = []
        current_byte = []
        for b in byte_seq:
            if b == 255:
                if current_byte:#有内容
                    text_parts.append(bytes(current_byte).decode('utf-8', errors='replace'))
                    current_byte = []
            else:
                current_byte.append(b)
        return ' '.join(text_parts)


# ------------------------------
# 测试：用简单文本训练和编码
# ------------------------------
if __name__ == "__main__":
    # 训练文本（简单示例，实际需用大规模语料）
    train_text = """apple banana apple pear orange grape
                    apple is red banana is yellow pear is green
                    I love eating apple and banana"""
    
    # 初始化并训练（词汇表大小设为 300，包含 256 字节 + 44 个合并子词）
    bpe = ByteLevelBPE(vocab_size=300)
    bpe.train(train_text)
    
    # 编码测试文本
    test_text = "I love apple and grape!"
    encoded = bpe.encode(test_text)
    print(f"\n测试文本：{test_text}")
    print(f"编码结果（前 10 个 ID）：{encoded[:10]}")
    
    # 解码验证
    decoded = bpe.decode(encoded)
    print(f"解码结果：{decoded}")