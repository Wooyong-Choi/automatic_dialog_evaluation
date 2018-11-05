import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.utils import pad_batch

class UnrefScorer(nn.Module):
    def __init__(self, dim, nhidden, nlayer, ntoken, nbatch, device):
        super(UnrefScorer, self).__init__()
        self.hidden = nhidden
        self.layer  = nlayer
        self.batch  = nbatch
        self.device = device

        self.src_embed = nn.Embedding(ntoken, dim)
        self.tar_embed = nn.Embedding(ntoken, dim)

        self.src_lstm  = nn.LSTM(dim, nhidden, nlayer, bidirectional=True)
        self.tar_lstm  = nn.LSTM(dim, nhidden, nlayer, bidirectional=True)
        
    def forward(self, src, src_len, tar, tar_len):

        h0 = torch.zeros(self.layer*2, len(src), self.hidden).to(self.device)
        c0 = torch.zeros(self.layer*2, len(src), self.hidden).to(self.device)

        padded_src_batch, src_batch_len, src_sorted_indices = pad_batch(src, src_len, self.device)
        padded_tar_batch, tar_batch_len, tar_sorted_indices = pad_batch(tar, tar_len, self.device)
        padded_src_batch = padded_src_batch.squeeze(-1)
        padded_tar_batch = padded_tar_batch.squeeze(-1) 

        _, src_original_indices = torch.sort(src_sorted_indices)
        _, tar_original_indices = torch.sort(tar_sorted_indices)

        src_emb = self.src_embed(padded_src_batch)
        tar_emb = self.tar_embed(padded_tar_batch)
        
        packed_src = pack_padded_sequence(src_emb, src_batch_len)
        packed_tar = pack_padded_sequence(tar_emb, tar_batch_len)
        
        output_src, (src_h, src_c) = self.src_lstm(packed_src, (h0, c0))
        output_tar, (tar_h, tar_c) = self.tar_lstm(packed_tar, (h0, c0))
        
        src_h, src_h_len = pad_packed_sequence(src_h)
        tar_h, tar_h_len = pad_packed_sequence(tar_h)
        #print('before', src_h.size())
        src_h = torch.cat((src_h[0,:,:], src_h[-1,:,:]), 1)
        tar_h = torch.cat((tar_h[0,:,:], tar_h[-1,:,:]), 1)
        #print('after', src_h.size())

        cat_h = torch.cat((src_h, tar_h), 1)
        #print('cat', cat_h.size())
        #torch.stack([src_h[idx] for idx in src_original_indices])
        #torch.stack([tar_h[idx] for idx in tar_original_indices])

        return cat_h

