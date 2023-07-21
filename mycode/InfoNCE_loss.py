import torch

import torch.nn as nn
from torch.nn import functional as F


def weighted_soft_margin_loss(diff, beta=10.0, reduction=torch.mean):
    out = torch.log(1 + torch.exp(diff * beta))
    if reduction:
        out = reduction(out)
    return out


def get_semi_hard_neg(logits, pos_dist):   
    N = logits.shape[0]
    targets = torch.arange(0, N, dtype=torch.long, device=logits.device)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask_= torch.lt(logits, pos_dist.unsqueeze(1))
    mask[targets, targets] = False
    mask_[targets, targets] = False

    mininum = torch.mul(logits, mask_)
    hard_neg_dist1, _ = torch.max(mininum, 1)
    hard_neg_dist2, _ = torch.min(logits[mask].reshape(N,-1), 1)
    hard_neg_dist = torch.max(hard_neg_dist1, hard_neg_dist2)

    return hard_neg_dist

def get_topk_hard_neg(logits, pos_dist, k):   
    N = logits.shape[0]
    targets = torch.arange(0, N, dtype=torch.long, device=logits.device)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[targets, targets] = False
    hard_neg_dist, _ = torch.topk(logits[mask].reshape(N,-1), largest=False, k=k, dim=1)
    return hard_neg_dist


class SemiSoftTriHard(nn.Module):
    '''
        Triplet loss with negative sample mining
        Modified from SAIG
    '''
    def __init__(self):
        super().__init__()
        
    def forward(self, grd_global, sat_global, args):
        dist_array = 2.0 - 2.0 * torch.matmul(sat_global, grd_global.t())
        pos_dist = torch.diag(dist_array)
        logits = dist_array
        #hard_neg_dist_g2s = get_semi_hard_neg(logits, pos_dist)
        #hard_neg_dist_s2g = get_semi_hard_neg(logits.t(), pos_dist)
        hard_neg_dist_g2s = get_topk_hard_neg(logits, pos_dist, int(args.train_batch_size**0.5))
        hard_neg_dist_s2g = get_topk_hard_neg(logits.t(), pos_dist, int(args.train_batch_size**0.5))
        return (weighted_soft_margin_loss(pos_dist - hard_neg_dist_g2s.t(), args.loss_weight) + 
                weighted_soft_margin_loss(pos_dist - hard_neg_dist_s2g.t(), args.loss_weight)) / 2.0


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of cosine similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
        git@github.com:RElbers/info-nce-pytorch.git

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction # corresponding output reduction of cross-entropy
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return self.info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)

    def info_nce(self, query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        # Check input dimensionality.
        if query.dim() != 2:
            raise ValueError('<query> must have 2 dimensions.')
        if positive_key.dim() != 2:
            raise ValueError('<positive_key> must have 2 dimensions.')
        if negative_keys is not None:
            if negative_mode == 'unpaired' and negative_keys.dim() != 2:
                raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
            if negative_mode == 'paired' and negative_keys.dim() != 3:
                raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

        # Check matching number of samples.
        if len(query) != len(positive_key):
            raise ValueError('<query> and <positive_key> must must have the same number of samples.')
        if negative_keys is not None:
            if negative_mode == 'paired' and len(query) != len(negative_keys):
                raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

        # Embedding vectors should have same number of components.
        if query.shape[-1] != positive_key.shape[-1]:
            raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
        if negative_keys is not None:
            if query.shape[-1] != negative_keys.shape[-1]:
                raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

        # Normalize to unit vectors
        query, positive_key, negative_keys = self.normalize(query, positive_key, negative_keys)
        if negative_keys is not None:
            # Explicit negative keys
            # Cosine between positive pairs
            positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

            if negative_mode == 'unpaired':
                # Cosine between all query-negative combinations
                negative_logits = query @ self.transpose(negative_keys)

            elif negative_mode == 'paired':
                query = query.unsqueeze(1)
                negative_logits = query @ self.transpose(negative_keys)
                negative_logits = negative_logits.squeeze(1)

            # First index in last dimension are the positive samples
            logits = torch.cat([positive_logit, negative_logits], dim=1)
            # at https://github.com/RElbers/info-nce-pytorch/issues/2#issuecomment-901384938
            labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        else:
            # Negative keys are implicitly off-diagonal positive keys.
            # Cosine between all combinations
            logits = query @ self.transpose(positive_key)
            # Positive keys are the entries on the diagonal
            labels = torch.arange(len(query), device=query.device)

        return F.cross_entropy(logits / temperature, labels, reduction=reduction)

    def transpose(self, x):
        return x.transpose(-2, -1)

    def normalize(self, *xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]


class InfoNCE_vanilla(nn.Module):
    '''
        This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
        A query embedding is compared with one positive key and without negative keys.
        Infact:
            Cross entropy loss with positive and negatives
        Input: 
            N,C query and positives in mini-batch, negative keys are implicitly off-diagonal positive keys. 
    '''
    def __init__(self, temperature=0.02):
        super().__init__()
        self.temperature=temperature

    def forward(self, pair1_feat, pair2_feat):
        # get batch size
        N = pair1_feat.shape[0]
        # feature similarity based on dot product
        logits = pair2_feat @ pair1_feat.t()
        # cat is a matrix trick? for get the similarity between query and positives on the diagonal
        # and negative keys are implicitly off-diagonal positive keys.
        logits = torch.cat((logits, logits.t()), dim=1)
        # positive keys are the entries on the diagonal
        # indicates the (0,0) and (1,1) in one_hot matrix are the true label 1, meanwhile 0 at other positions
        # supposed to be a N [0,1] tensor and one_hot coding in cross-entropy
        targets = torch.arange(0, N, dtype=torch.long, device=logits.device)
        mask = torch.ones_like(logits, dtype=torch.bool)
        # filter out the repeat similarity of query and positive from cat operation
        mask[targets, targets + N] = False
        # the final logits are the 
        logits = logits[mask].reshape(N, -1)
        return F.cross_entropy(logits/self.temperature, targets)



if __name__=="__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")  
    parser.add_argument("--loss_weight", default=10, type=float,
                        help="loss_weight")
    args = parser.parse_args()
    import random
    random.seed(1)
    torch.manual_seed(1)

    loss=InfoNCE(negative_mode = 'unpaired')
    # loss=InfoNCE_vanilla()
    # normalization is necessary
    grd_global = F.normalize(torch.rand(2, 2), dim=1)
    sat_global1 = F.normalize(torch.rand(2, 2), dim=1)
    sat_global2 = F.normalize(torch.rand(2, 2), dim=1)
    print(loss(grd_global, sat_global1, sat_global2))
    