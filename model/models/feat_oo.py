import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.models import FewShotModel


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head  # 1
        self.d_k = d_k  # 640
        self.d_v = d_v  # 640

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)  # ConvNet(64, 1*64)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))  # w_qs.weight:[640,640]
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()  # 1,5,640
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q  # [1,5,640]
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)  # [1,5,1,640]
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk [1,5,640]
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)  # [1, 1, 5, 640]
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv) [1,5,640]

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


class FEAT(FewShotModel):
    def __init__(self, args):
        super().__init__(args)
        if args.backbone_class == 'ConvNet':
            hdim = 64
        elif args.backbone_class == 'Res12':
            hdim = 640
        elif args.backbone_class == 'Res18':
            hdim = 512
        elif args.backbone_class == 'WRN':
            hdim = 640
        else:
            raise ValueError('')

        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.1)

    def _forward(self, instance_embs, support_idx, query_idx):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(
            *(support_idx.shape + (-1,)))  # [1,1,5,-1]->[1,1,5,640]   #???instance_embs???5???
        query = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(
            *(query_idx.shape + (-1,)))  # [1,15,5,640] ???instance_embs???5~80??????1-shot)???

        # get mean of the support
        proto = support.mean(dim=1)  # Ntask x NK x d  [1,1,5,640] ??????5-way 5-shot [1,5,5,640]-->?????????[1,5,640]
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]  # 5
        num_query = np.prod(query_idx.shape[-2:])  # 75

        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        proto = self.slf_attn(proto, proto, proto)
        if self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1)  # (Nbatch*Nq*Nw, 1, d)  [75,1,640]
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            proto = proto.view(num_batch * num_query, num_proto, emb_dim)  # (Nbatch x Nq, Nk, d) [75,5,640]

            logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
        else:
            proto = F.normalize(proto, dim=-1)  # normalize for cosine distance  (1,5,640)
            query = query.view(num_batch, -1, emb_dim)  # (Nbatch,  Nq*Nw, d)   (1,75,640)

            logits = torch.bmm(query, proto.permute([0, 2, 1])) / self.args.temperature
            logits = logits.view(-1, num_proto)  # (75,5)

        # for regularization
        if self.training:
            aux_task = torch.cat([support.view(1, self.args.shot, self.args.way, emb_dim),
                                  query.view(1, self.args.query, self.args.way, emb_dim)], 1)  # T x (K+Kq) x N x d
            num_query = np.prod(aux_task.shape[1:3])
            aux_task = aux_task.permute([0, 2, 1, 3])  # [1,5,16,640]
            aux_task = aux_task.contiguous().view(-1, self.args.shot + self.args.query,
                                                  emb_dim)  # (5,16,64) //[5,20,640]
            # apply the transformation over the Aug Task
            aux_emb = self.slf_attn(aux_task, aux_task, aux_task)  # T x N x (K+Kq) x d
            # compute class mean
            aux_emb = aux_emb.view(num_batch, self.args.way, self.args.shot + self.args.query, emb_dim)  # ???1???5???20???640???
            aux_center = torch.mean(aux_emb, 2)  # T x N x d   (1,5,64)  [

            if self.args.use_euclidean:
                aux_task = aux_task.permute([1, 0, 2]).contiguous().view(-1, emb_dim).unsqueeze(
                    1)  # (Nbatch*Nq*Nw, 1, d)  [100,1,640]
                aux_center = aux_center.unsqueeze(1).expand(num_batch, num_query, num_proto,
                                                            emb_dim).contiguous()  # [1,100,5,640]
                aux_center = aux_center.view(num_batch * num_query, num_proto,
                                             emb_dim)  # (Nbatch x Nq, Nk, d) [100,5,640]

                logits_reg = - torch.sum((aux_center - aux_task) ** 2, 2) / self.args.temperature2
            else:
                aux_center = F.normalize(aux_center, dim=-1)  # normalize for cosine distance   [1,5,640]
                aux_task = aux_task.permute([1, 0, 2]).contiguous().view(num_batch, -1,
                                                                         emb_dim)  # (Nbatch,  Nq*Nw, d)   [1,80,640]

                logits_reg = torch.bmm(aux_task, aux_center.permute([0, 2, 1])) / self.args.temperature2
                logits_reg = logits_reg.view(-1, num_proto)

            return logits, logits_reg
        else:
            return logits
