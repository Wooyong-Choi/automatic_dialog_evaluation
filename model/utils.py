import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from onmt.modules import Embeddings
from onmt.encoders import RNNEncoder

def collate_fn(batch):
    """
    Collate function for DataLoader class in Pytorch
    """
    #print ('collate', batch)
    return list(zip(*batch))


def pad_batch(indices, lengths, device):
    """
    Pad inputs of a model.
    
    Return:
        Padded batch data (Decreasing order sorted)
        Sorted batch length
        Sorted indices (for recovering original order)
    """
    length_batch = torch.tensor(lengths, dtype=torch.long, device=device)
    sorted_length, sorted_indices = torch.sort(length_batch, descending=True)
    
    padded_batch = nn.utils.rnn.pad_sequence(
        [torch.tensor(indices[idx], dtype=torch.long, device=device) for idx in sorted_indices]
    )
    padded_batch = padded_batch.unsqueeze(2)
    return padded_batch, sorted_length, sorted_indices


def load_fields_from_vocab(pre_vocab_path):    
    """
    Load prepocessed vocabulary dictionary in onmt Field objects from `vocab.pt` file.
    It is necessory for load pretrained word embedding layer for encoder model on onmt.
    """
    onmt_vocab = dict(torch.load(pre_vocab_path))
    
    # Source 단의 Encoder와 Pretrained Embedding을 가져올 것이므로
    # Source vocab dictionary만 사용함
    n_words = len(onmt_vocab['src'])
    index2word = onmt_vocab['src'].itos
    word2index = onmt_vocab['src'].stoi
    
    return n_words, index2word, word2index
    

def build_pretrained_model(model_path, vocab):
    """
    Load pretrained model of onmt.
    Args:
        model_path (string)
        vocab (Vocab) : words dictionary to build embedding layer
    """
    
    def build_embeddings(vocab, embed_dim):
        """
        Build an Embeddings instance.
        Args:
            opt: the option in current environment.
            vocab(Vocab): words dictionary.
        """        
        embedding_dim = embed_dim
        word_padding_idx = vocab.pad_idx
        num_word_embeddings = len(vocab)
    
        return Embeddings(word_vec_size=embedding_dim,
                          word_padding_idx=word_padding_idx,
                          word_vocab_size=num_word_embeddings)
    
    def build_encoder(num_layers, rnn_size, embeddings, bidirectional=True):
        """
        Build an Encoder instance.
        Args:
            num_layers (int)
            rnn_size (int)
            bidirectional (bool)
            embeddings (Embeddings): vocab embeddings for this encoder.
        """
        return RNNEncoder(
            rnn_type="GRU",
            bidirectional=bidirectional,
            num_layers=num_layers,
            embeddings=embeddings,
            hidden_size=rnn_size
        )
    
    # Exmaple keys to get model info
    encoder_key = 'encoder.'
    bidir_key = 'reverse'
    embed_example_key = 'embeddings.make_embedding.emb_luts.0.weight'
    rnn_example_key = 'rnn.weight_hh_l0'
    
    print("Loading pretrained model... \n")
    model_params = torch.load(model_path)
    
    # Extract only the parameters of encoder
    encoder_weight_dict = {k[8:]:v for k, v in model_params['model'].items() if k.startswith(encoder_key)}
    
    # Get model infomation
    bidirectional = any(keys for keys in encoder_weight_dict.keys() if keys.endswith(bidir_key))
    num_direction = 2 if bidirectional else 1
    # (전체 weight 개수 - 1 (Embedding layer weight)) / 2 (ih, hh) / 2 (weight, bias) / num_direction
    num_layers = int((len(encoder_weight_dict)-1) / 2 / 2 / num_direction)
    input_size, embed_size = encoder_weight_dict[embed_example_key].size()
    rnn_size = encoder_weight_dict[rnn_example_key].size(1) * num_direction
    
    print(num_layers)
    
    # Check whether a proper pair of vocab dict and pretrained embedding layer
    assert vocab.n_words == input_size
    
    # Build a model
    embeddings = build_embeddings(vocab, embed_size)
    encoder = build_encoder(num_layers, rnn_size, embeddings=embeddings, bidirectional=bidirectional)
<<<<<<< HEAD
    #print(encoder, '\n')
=======
    encoder.load_state_dict(encoder_weight_dict)
    print(encoder, '\n')
>>>>>>> refs/remotes/origin/master
    
    return encoder  

class NegativeDataset(Dataset):
    def __init__(self, dataset, vocab):
        super(Dataset, self).__init__()

        print('[*] generate negative sampels')
        negative = list(zip(*dataset))
        target = list(negative[1])
        random.shuffle(target)
        negative_dataset = [list(negative[0]), target]
        self.data = list(zip(*negative_dataset))
        self.vocab = vocab
        print('Negative sample size: ', len(self.data))
        print('Negative example: ', self.data[0])
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item_list = []
        for item in self.data[index]:
            item_list.append(self.vocab.sentence_to_indices(item))
            item_list.append(len(item))
        return item_list
