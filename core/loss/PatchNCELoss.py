from packaging import version
import torch
from torch import nn
from math import *
import torch.nn.functional as F

class PatchNCELoss(nn.Module):
    def __init__(self, device='cuda:0'):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.device = device

    def forward(self, feat_fv, feat_fi, feat_v, feat_i):
        batchSize = feat_fv.shape[0]

        dim = feat_fv.shape[1]
        T = 0.07

        feat_fv = F.normalize(feat_fv, dim=1)
        feat_fi = F.normalize(feat_fi, dim=1)
        feat_v = F.normalize(feat_v, dim=1)
        feat_i = F.normalize(feat_i, dim=1)


        l_pos_v = torch.bmm(feat_fv.view(batchSize, 1, -1), feat_v.view(batchSize, -1, 1))
        l_pos_v = l_pos_v.view(batchSize, 1)

        l_pos_i = torch.bmm(feat_fi.view(batchSize, 1, -1), feat_i.view(batchSize, -1, 1))
        l_pos_i = l_pos_i.view(batchSize, 1)



        l_pos = torch.zeros((batchSize, 1)).to(self.device)
        l_neg = torch.zeros((batchSize, batchSize-1)).to(self.device)

        for b in range(batchSize):

            if l_pos_v[b] >= l_pos_i[b]:
                # pos logit
                l_pos_batch = l_pos_v[b].unsqueeze(0)
                l_pos = l_pos.scatter_add_(0, torch.tensor([[b]]).to(self.device), l_pos_batch)

                # neg logit
                feat_v_without_pos = feat_v[torch.arange(feat_v.size(0)) != b]
                index_base = [b for _ in range(batchSize-1)]
                index = torch.tensor([index_base]).to(self.device)
                l_neg_batch = torch.bmm(feat_fv[b].unsqueeze(0).unsqueeze(0), feat_v_without_pos.view(1, -1, dim).transpose(2, 1))
                l_neg_batch = l_neg_batch.view(-1, batchSize-1)
                l_neg = l_neg.scatter_add_(0, index, l_neg_batch)
            else:
                # pos logit
                l_pos_batch = l_pos_i[b].unsqueeze(0)
                l_pos = l_pos.scatter_add_(0, torch.tensor([[b]]).to(self.device), l_pos_batch)

                # neg logit
                feat_i_without_pos = feat_i[torch.arange(feat_i.size(0)) != b]
                index_base = [b for _ in range(batchSize-1)]
                index = torch.tensor([index_base]).to(self.device)
                l_neg_batch = torch.bmm(feat_fi[b].unsqueeze(0).unsqueeze(0), feat_i_without_pos.view(1, -1, dim).transpose(2, 1))
                l_neg_batch = l_neg_batch.view(-1, batchSize-1)
                l_neg = l_neg.scatter_add_(0, index, l_neg_batch)

        out = torch.cat((l_pos, l_neg), dim=1) / T
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_fv.device))

        return loss
