import operator

import torch

from .utils import load_fields_from_vocab

class Vocab:
    """
    A vocabulary class from onmt preprocessed vocab fields.
    """
    
    def __init__(self, pre_vocab_path):
        # Load onmt vocab dict
        print('[*] Loading onmt vocab dictionary...')
        self.n_words, self.index2word, self.word2index = load_fields_from_vocab(pre_vocab_path)
        
        self.unk_tok = self.index2word[0]
        self.pad_tok = self.index2word[1]
        self.unk_idx = 0
        self.pad_idx = 1
        
        print('Number of dictionary : {} \n'.format(self.n_words))
    
    def sentence_to_indices(self, sentence):
        return [self.word2index[word] if word in self.word2index else self.unk_idx for word in sentence]
    
    def indices_to_sentence(self, indices):
        return [self.index2word[idx] if idx in self.index2word else self.unk_tok for idx in indices]
    
    def __len__(self):
        return self.n_words
    
    