import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.models import FewShotModel


class ScaledDotProductAttention0(nn.Module):
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


class MultiHeadAttention0(nn.Module):
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

        self.attention = ScaledDotProductAttention0(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
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


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        # q:[1,25,640]  k.T [1,640,75]  for 5-shot
        # 假定每个任务里，查询取15; 加的临时变量，可以用args里值
        nb = q.shape[0]
        n_way = 5  # 5
        n_query = 15
        n_shot = int(q.shape[1] / 5)  # 1  / 5
        dim = q.shape[-1]

        attn = torch.bmm(q, k.transpose(1, 2))  # [n,k,q]   [1,5,75] /[1,75,5] //[1,75,640]*[1,5,640]^T-->[1,75,5]
        attn = attn / self.temperature  # [1,25,75]
        # attn = self.softmax(attn)
        attn_avg = attn.contiguous().view(nb, -1, n_way, k.shape[1]).mean(
            dim=1)  # [n,n_way,shot,q] 求平均 [1,5/15,5,75]->[1,5,75]
        # log_attn = F.log_softmax(attn, 2)   #要修改， attn-->attn_avg
        attn_avg = self.softmax(attn_avg)  # [n,k,q]
        attn_avg = self.dropout(attn_avg)
        proto = torch.bmm(attn_avg, v)  # [n,k,m]   proto_q; [n,5,640]

        return proto, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k  # [640]
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)  # ConvNet(64, 1*64)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(n_head * d_model, d_model, bias=False)
        nn.init.xavier_normal_(self.fc.weight)
        self.relu = nn.LeakyReLU(0.1)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()  # q.size; [1,25,640]   sz_b=1; len_q=5; len_k=len_v=75;// len_q=75; len_k=len_v=5;
        sz_b, len_k, _ = k.size()  # k.size: [1,75,640]   //
        sz_b, len_v, _ = v.size()

        # residual = q
        q = self.w_qs(q).view(sz_b, len_q, d_k)  # view(1,25,640) // [1,75,640]
        k = self.w_ks(k).view(sz_b, len_k, d_k)  # [1,75,640]  //[1,25,640]
        v = self.w_vs(v).view(sz_b, len_v, d_v)
        #
        # q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q,
        #                                             d_k)  # (n*b) x lq x dk   对5-way 1-shot:[1,5,640]   //[1,75,640]
        # k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk           [1,75,640]  //[1,5,640]
        # v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        dim = q.shape[-1]
        way = num_proto = 5
        output, attn = self.attention(q, k,
                                      v)  # 相当于原型[1,5,640]
        # output = output.contiguous().view(sz_b, way, -1, dim)  # [1,5,1,640]
        # k = k.contiguous().view(sz_b, way, -1, dim)  # [1,75,640]
        output = self.dropout(output)
        output = output.unsqueeze(1).expand(output.shape[0], int(k.shape[1] / 5), num_proto, dim).contiguous()
        output = output.contiguous().view(k.shape)  # [1,75,640]
        output = torch.add(output, k)  # [1,75,640]  //[1,25,640]
        output = self.layer_norm1(output)
        residual = output
        output = self.relu(self.fc(output))
        # output = self.dropout(output)
        output += residual
        output = self.layer_norm2(output)

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
        n_way = args.way
        n_shot = args.shot
        n_query = args.query

        # self.slf_attn0 = MultiHeadAttention0(1, hdim, hdim, hdim, dropout=0.1)
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.1)

    def _forward(self, instance_embs, support_idx, query_idx):
        emb_dim = instance_embs.size(-1)
        # support_idx:[1,5,5];
        # query_idx:[1,15,5]
        # organize support/query data
        # 从instance_embs[]取前5个，-->根据idx[1,5,5]加上最后一维[1,5,5,640] （5-shot)
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(
            *(support_idx.shape + (-1,)))  # [1,1,5,640]
        # instance_embs[]后面75个作为query
        query = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(
            *(query_idx.shape + (-1,)))  # [1,15,5,640]
        num_batch = support_idx.shape[0]  # 1
        num_way = 5  # 5
        num_support = np.prod(support_idx.shape[-2:])  # shot: 1*5 / 5*5
        num_query = np.prod(query_idx.shape[-2:])  # 15*5

        support = support.contiguous().view(num_batch, -1, emb_dim)  # [1,5,5,640]-->[1,25,640]
        query = query.contiguous().view(num_batch, -1, emb_dim)  # [1,5,15,640]-->[1,75,640]

        z_query = self.slf_attn(support, query, query)  # [1,75,640]
        z_support = self.slf_attn(query, support, support)  #:[1,25,640]
        z_support = z_support.contiguous().view(num_batch, -1, num_way, emb_dim)  # 【1，5，5，640】
        # get mean of the support
        proto = z_support.mean(dim=1)  # Ntask x NK x d     #[1,5,640]
        num_proto = proto.shape[1]

        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        if self.args.use_euclidean:
            z_query = z_query.view(-1, emb_dim).unsqueeze(1)  # (Nbatch*Nq*Nw, 1, d)  [75,1,640]
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()  # [1,75,5,640]
            proto = proto.view(num_batch * num_query, num_proto, emb_dim)  # (Nbatch x Nq, Nk, d)[75,5,640]

            logits = - torch.sum((proto - z_query) ** 2, 2) / self.args.temperature
        else:
            # 余弦距离用在正空间，将两个值到(0,1)之间。
            proto = F.normalize(proto, dim=-1)  # normalize for cosine distance  (1,5,640)
            z_query = z_query.view(num_batch, -1, emb_dim)  # (Nbatch,  Nq*Nw, d)   (1,75,640)

            logits = torch.bmm(z_query, proto.permute([0, 2, 1])) / self.args.temperature
            logits = logits.view(-1, num_proto)  # (75,5)

        # for regularization
        if self.training:
            aux_task = torch.cat([z_support.view(1, self.args.shot, self.args.way, emb_dim),
                                  z_query.view(1, self.args.query, self.args.way, emb_dim)],
                                 1)  # T x (K+Kq) x N x d [1,16,5,640]
            num_query = np.prod(aux_task.shape[1:3])  # 80
            # aux_task = aux_task.permute([0, 2, 1, 3])
            # apply the transformation over the Aug Task
            # aux_emb = self.slf_attn(aux_task, aux_task, aux_task) # T x N x (K+Kq) x d
            # compute class mean; 没有加slf_attn
            # aux_emb = aux_task.view(num_batch, self.args.way, self.args.shot + self.args.query, emb_dim)  #[1,5,16,640]
            aux_center = torch.mean(aux_task, 1)  # T x N x d    [1,5,640]
            aux_task = aux_task.contiguous().view(-1, emb_dim)  # (80,640)

            if self.args.use_euclidean:
                aux_task = aux_task.unsqueeze(1)  # (Nbatch*Nq*Nw, 1, d)[80,1,640]
                aux_center = aux_center.unsqueeze(1).expand(num_batch, num_query, num_proto,
                                                            emb_dim).contiguous()  # [1,80,5,640]
                aux_center = aux_center.view(num_batch * num_query, num_proto,
                                             emb_dim)  # (Nbatch x Nq, Nk, d) [100,5,640]

                logits_reg = - torch.sum((aux_center - aux_task) ** 2, 2) / self.args.temperature2
            else:
                aux_center = F.normalize(aux_center, dim=-1)  # normalize for cosine distance
                aux_task = aux_task.contiguous().view(num_batch, -1, emb_dim)  # (Nbatch,  Nq*Nw, d) [1,80,640]

                logits_reg = torch.bmm(aux_task, aux_center.permute([0, 2, 1])) / self.args.temperature2  # [1,80,5]
                logits_reg = logits_reg.view(-1, num_proto)

            return logits, logits_reg
        else:
            return logits
