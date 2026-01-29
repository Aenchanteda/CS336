from collections import Counter
import regex as re


def _gpt2_tokenizer_pattern():
	return re.compile(
		r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
		re.UNICODE,
	)


class BPETokenizer:
	def __init__(self, num_merges, special_tokens=None):
		self.num_merges = num_merges
		self.special_tokens = list(special_tokens or [])
		self.pattern = _gpt2_tokenizer_pattern()
		self.vocab = Counter()
		self.merges = []
		self.token_to_id = {}
		self.id_to_token = []

	def _pre_tokenize(self, text):
		return self.pattern.findall(text)

	def _build_vocab_from_docs(self, docs):
		self.vocab.clear()
		for doc in docs:
			self.vocab.update(doc)
		for token in self.special_tokens:
			self.vocab[token] += 1

	def _pair_counts(self, docs):
		counts = Counter()
		for doc in docs:
			for i in range(len(doc) - 1):
				counts[(doc[i], doc[i + 1])] += 1
		return counts

	def _merge_tokens(self, tokens, pair):
		merged = []
		i = 0
		while i < len(tokens):
			if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
				merged.append(tokens[i] + tokens[i + 1])
				i += 2
			else:
				merged.append(tokens[i])
				i += 1
		return merged

	def _apply_merges(self, tokens):
		current = list(tokens)
		for pair in self.merges:
			current = self._merge_tokens(current, pair)
		return current

	def _finalize_vocab(self):
		ordered_tokens = []
		seen = set()
		for token in self.special_tokens:
			if token not in seen:
				ordered_tokens.append(token)
				seen.add(token)
		for token, _ in self.vocab.most_common():
			if token not in seen:
				ordered_tokens.append(token)
				seen.add(token)
		self.id_to_token = ordered_tokens
		self.token_to_id = {token: idx for idx, token in enumerate(ordered_tokens)}

	def train(self, texts):
		docs = [self._pre_tokenize(text) for text in texts]
		self._build_vocab_from_docs(docs)

		while len(self.merges) < self.num_merges:
			pair_counts = self._pair_counts(docs)
			if not pair_counts:
				break
			pair, _ = pair_counts.most_common(1)[0]
			self.merges.append(pair)
			docs = [self._merge_tokens(doc, pair) for doc in docs]
			self._build_vocab_from_docs(docs)

		self._finalize_vocab()

	def encode(self, text):
		tokens = self._pre_tokenize(text)
		tokens = self._apply_merges(tokens)
		ids = []
		for token in tokens:
			if token in self.token_to_id:
				ids.append(self.token_to_id[token])
			else:
				ids.append(self.token_to_id.get(token, -1))
		return ids

	def decode(self, ids):
		pieces = []
		for idx in ids:
			if 0 <= idx < len(self.id_to_token):
				pieces.append(self.id_to_token[idx])
		return "".join(pieces)
