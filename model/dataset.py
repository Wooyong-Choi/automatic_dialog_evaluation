import re
import sys
import random
from operator import itemgetter
from torch.utils.data import Dataset
from model.vocab import Vocab

class Dataset(Dataset):
    """
    A dataset basically supports iteration over all the examples it contains.
    We currently supports only text data with this class.
    This class is inheriting Dataset class in torch.utils.data.
    """

    def __init__(self, vocab_path, data_path_list, max_length):
        super(Dataset, self).__init__()
        
        self.vocab_path = vocab_path
        self.data_path_list = data_path_list    
        self.max_length = max_length
        self.data = None
        self.vocab = Vocab(self.vocab_path)
        self._prepareData()
        
    def __getitem__(self, index):
        item_list = []
        for item in self.data[index]:
            item_list.append(self.vocab.sentence_to_indices(item))
            item_list.append(len(item))
        return item_list
                
    def __len__(self):
        return len(self.data)
    
    def _prepareData(self):
        data = self._readData()
        print("Read {} sentence pairs".format(len(data)))
        
        data = self._filterDatas(data)
        print("Trim data to {} sentence pairs \n".format(len(data)))
        
        print("[*] Success to preprocess data! \n")
        
        self.data = data

    def _readData(self):
        print("[*] Reading lines...")
    
        # Read the file and split into lines
        lines_list = [[self._preprocessing(l).split(' ') for l in open(file_path, 'r', encoding='utf-8').readlines()]
                      for file_path in self.data_path_list]
        data = list(zip(*lines_list))
        
        # Print statistics
        for i, lines in enumerate(lines_list):
            print("Avg length of data {} : {:.2f}".format(i, sum([len(l) for l in lines]) / len(data)))
        print()
        
        return data
    
    def _preprocessing(self, s):
        return s.strip().lower()
    
    def _filterDatas(self, data):
        data = [d for d in data if self._chkMaxLength(d)]
        return data

    def _chkMaxLength(self, p):
        return len(p[0]) <= self.max_length and len(p[1]) <= self.max_length and len(p[0]) > 0 and len(p[1]) > 0

    def getInstanceSize(self):
        return len(self.data)

    def getVocabSize(self):
        return self.vocab.__len__()

