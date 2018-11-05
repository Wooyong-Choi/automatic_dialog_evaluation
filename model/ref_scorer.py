import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import pad_batch

class RefScorer(object):
    def __init__(self, pretrained_encoder, device='cpu'):
        super(RefScorer, self).__init__()
        self.encoder = pretrained_encoder
        self.device = device
        
        self.encoder.to(self.device)
        
    def get_ref_score(self, gold_batch, gold_len, gen_batch, gen_len):
        """
        Calculate reference scores with gold sequence batch and generated sequence batch
        
        Return:
            Cosine similarities [Batch_size]
        """
        gold_emb = self._embed_sentence(gold_batch, gold_len)
        gen_emb = self._embed_sentence(gen_batch, gen_len)
        
        cos_sim = F.cosine_similarity(gold_emb, gen_emb)
        return cos_sim
    
    def _embed_sentence(self, batch, batch_len):
        padded_batch, batch_len, sorted_indices = pad_batch(batch, batch_len, self.device)
        _, original_indices = torch.sort(sorted_indices)
        
        embeded, _ = self.encoder(padded_batch, batch_len)
        embeded = torch.cat([e for e in embeded], dim=1)
        
        # Recover original index
        embeded = torch.stack([embeded[idx] for idx in original_indices])
        return embeded
