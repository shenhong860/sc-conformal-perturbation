"""Small torchtext.vocab replacement used by scGPT's GeneVocab."""

from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional


class Vocab:
    def __init__(
        self,
        vocab_dict=None,
        max_size=None,
        min_freq=1,
        specials=None,
        special_first=True,
        default_index=None,
        vectors=None,
        unk_init=None,
        vectors_cache=None,
    ):
        self.vocab = OrderedDict(vocab_dict if vocab_dict is not None else {})
        self.default_index = default_index

    def __len__(self):
        return len(self.vocab)

    def __contains__(self, token):
        return token in self.vocab

    def __getitem__(self, token):
        if token in self.vocab:
            return self.vocab[token]
        if self.default_index is not None:
            return self.default_index
        raise KeyError(token)

    def set_default_index(self, index):
        self.default_index = index

    def insert_token(self, token, index):
        self.vocab[token] = index

    def get_stoi(self):
        return dict(self.vocab)

    def get_itos(self):
        return list(self.vocab.keys())

    def lookup_indices(self, tokens):
        return [self[token] for token in tokens]

    def lookup_tokens(self, indices):
        itos = self.get_itos()
        return [itos[i] if i < len(itos) else "<unk>" for i in indices]


def vocab(ordered_dict, min_freq=1, specials=None, special_first=True):
    """Build a Vocab from a token->frequency OrderedDict."""
    tokens = [tok for tok, freq in ordered_dict.items() if freq >= min_freq]
    return Vocab({tok: idx for idx, tok in enumerate(tokens)})


def build_vocab_from_iterator(iterator, min_freq=1, specials=None, special_first=True):
    counter = Counter()
    for item in iterator:
        if isinstance(item, str):
            counter[item] += 1
        else:
            counter.update(item)
    ordered = OrderedDict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))
    return vocab(ordered, min_freq=min_freq, specials=specials, special_first=special_first)
