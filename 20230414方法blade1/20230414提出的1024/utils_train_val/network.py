import torch
import math
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .config import parse_args
from .common import Alignment, RandomJitter, set_seed

args = parse_args()
set_seed(SEED=2023)

class DGCNN(nn.Module):
    def __init__(self, emb_dims=512):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(6+args.neighbour_num, 64, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv1.weight, gain=1.0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv2.weight, gain=1.0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv3.weight, gain=1.0)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv4.weight, gain=1.0)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv5.weight, gain=1.0)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)

    def forward(self, x, if_relu_atlast=True): # x: batch*3*point_num
        batch_size, num_dims, num_points = x.size() # batch*3*point_num

        x = get_graph_feature(x)  # batch*dim*point_num*k

        x = F.relu(self.conv1(x))  # batch*dim*point_num*k
        x1 = x.max(dim=-1, keepdim=True)[0]  # batch*dim*point_num*1

        x = F.relu(self.conv2(x))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.conv3(x))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.conv4(x))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)  # batch*dim*point_num*1
        if if_relu_atlast == False:
            x = self.conv5(x).view(batch_size, -1, num_points) # batch*dim*point_num
        else:
            x = F.relu(self.conv5(x)).view(batch_size, -1, num_points) # batch*dim*point_num
        return x # batch*dim*point_num

def get_graph_feature(x, k=args.neighbour_num): # x: batch*3*point_num
    idx = knn(x, k=k)  # idx: (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    _, num_dims, _ = x.size()

    #   batch_size * num_points * k + range(0, batch_size*num_points)
    idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    idx = idx + idx_base.cuda()
    idx = idx.view(-1)

    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) # batch*point_num*k*3
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) # batch*point_num*k*3

    feature = torch.cat((x, feature, (((feature-x)**2).sum(-1).unsqueeze(-2).repeat(1, 1, k, 1))**0.5), dim=-1).permute(0, 3, 1, 2) # batch*(6+k)*point_num*k

    return feature

def knn(x, k): # x: batch*3*point_num
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx # idx: (batch_size, num_points, k)





class Transformer_self_attn(nn.Module):
    def __init__(self, args):
        super(Transformer_self_attn, self).__init__()
        self.emb_dims = args.emb_dims
        self.N = args.n_blocks
        self.dropout = args.dropout
        self.ff_dims = args.ff_dims
        self.n_heads = args.n_heads
        self.embed = nn.Sequential()
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(d_model=self.emb_dims, d_ff=self.ff_dims)
        self.self_attn = Self_attn(layer=self_attn_layer(size=self.emb_dims, self_attn=c(attn), feed_forward=c(ff)), N=self.N)

    def forward(self, input):
        input = input.transpose(2, 1).contiguous() # batch*dim*point_num--->batch*point_num*dim
        embedding_self_attn = self.self_attn(self.embed(input)).transpose(2, 1).contiguous()
        return embedding_self_attn # batch*dim*point_num

class Transformer_cross_attn(nn.Module):
    def __init__(self, args):
        super(Transformer_cross_attn, self).__init__()
        self.emb_dims = args.emb_dims
        self.N = args.n_blocks
        self.dropout = args.dropout
        self.ff_dims = args.ff_dims
        self.n_heads = args.n_heads
        self.view_a_embed = nn.Sequential()
        self.view_b_embed = nn.Sequential()
        self.generator = nn.Sequential()
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(d_model=self.emb_dims, d_ff=self.ff_dims)
        self.cross_attn = Cross_attn(layer=cross_attn_layer(size=self.emb_dims, cross_attn=c(attn), feed_forward=c(ff)), N=self.N)

    def forward(self, view_a, view_b):
        view_a = view_a.transpose(2, 1).contiguous() # batch*dim*point_num--->batch*point_num*dim
        view_b = view_b.transpose(2, 1).contiguous()

        view_a_embedding_cross_attn = self.cross_attn(x=self.view_a_embed(view_a), memory=view_b)
        view_a_embedding_cross_attn = self.generator(view_a_embedding_cross_attn).transpose(2, 1).contiguous()
        return view_a_embedding_cross_attn # batch*dim*point_num

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value):
        "Implements Figure 2"
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x = attention(query, key, value)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x) # batch*point_num*dim

def attention(query, key, value):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.BatchNorm1d(d_ff)  # nn.Sequential()
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())

class self_attn_layer(nn.Module):
    def __init__(self, size, self_attn, feed_forward):
        super(self_attn_layer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size), 2)
        self.size = size

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x)) # batch*point_num*dim
        return self.sublayer[1](x, self.feed_forward)

class cross_attn_layer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, cross_attn, feed_forward):
        super(cross_attn_layer, self).__init__()
        self.size = size
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size), 2)
        self.norm = LayerNorm(size)

    def forward(self, x, memory):
        "Follow Figure 1 (right) for connections."
        memory = self.norm(memory)
        x = self.sublayer[0](x, lambda x: self.cross_attn(x, memory, memory))
        return self.sublayer[1](x, self.feed_forward)

class SublayerConnection(nn.Module):
    def __init__(self, size):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Self_attn(nn.Module):
    def __init__(self, layer, N):
        super(Self_attn, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class Cross_attn(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Cross_attn, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory):
        for layer in self.layers:
            x = layer(x, memory)
        return self.norm(x)





class MLPNet(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
        super().__init__()
        list_layers = mlp_layers(nch_input, nch_layers, b_shared, bn_momentum, dropout)
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp):
        out = self.layers(inp)
        return out

def mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
    """ [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
            init.xavier_normal_(weights.weight, gain=1.0)
        else:
            weights = torch.nn.Linear(last, outp)
            init.xavier_normal_(weights.weight, gain=1.0)
        layers.append(weights)
#        layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
        layers.append(torch.nn.ReLU())
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers




def symfn_avg(x):
    a = torch.nn.functional.avg_pool1d(x, x.size(-1))
    #a = torch.sum(x, dim=-1, keepdim=True) / x.size(-1)
    return a

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        # Basic settings
        self.emb_dims = args.emb_dims
        self.state_dim = args.emb_dims

        ######################## LAYERS #########################
        ############ The layers for Init ############
        ### 1. Init the state H
        self.init_emb_nn_state = DGCNN(emb_dims=self.state_dim)
        ### 1. Feature extraction
        self.init_emb_nn = DGCNN(emb_dims=self.emb_dims)
        ### 2. Transformer
        self.init_self_attn = Transformer_self_attn(args=args)
        self.init_cross_attn = Transformer_cross_attn(args=args)
        ### 3. point-wise weight
        mlp_w, mlp_w_2 = [512, 256, 256, 256], [64]
        self.init_weight_mlp = MLPNet(self.state_dim, mlp_w, b_shared=True).layers

        self.init_weight_mlp_2 = MLPNet(256, mlp_w_2, b_shared=True).layers
        self.init_weight = torch.nn.Conv1d(64, 2, 1)
        init.xavier_normal_(self.init_weight.weight, gain=1.0)

        ### 4. Rotary center
        mlp_or, mlp_or_2, mlp_or_3 = [512, 256], [256, 128, 64], [64, 64, 64]
        self.init_or_mlp = MLPNet(self.state_dim, mlp_or, b_shared=True).layers
        self.init_or_mlp_2 = MLPNet(512, mlp_or_2, b_shared=True).layers
        self.init_or_mlp_3 = MLPNet(64, mlp_or_3, b_shared=True).layers

        self.init_or_ = torch.nn.Conv1d(64, 2, 1)
        init.xavier_normal_(self.init_or_.weight, gain=1.0)
        ### The layers of GRU
        ### 1. Acquire the Z; 2. Acquire the R; 3. Acquire the H-wave
        self.init_mlp_z = torch.nn.Conv1d(self.emb_dims + (args.score_top_k + self.emb_dims) * (args.view_num - 1) + self.state_dim, self.state_dim, 1)
        self.init_mlp_r = torch.nn.Conv1d(self.emb_dims + (args.score_top_k + self.emb_dims) * (args.view_num - 1) + self.state_dim, self.state_dim, 1)
        self.init_mlp_hwave = torch.nn.Conv1d(self.emb_dims + (args.score_top_k + self.emb_dims) * (args.view_num - 1) + self.state_dim, self.state_dim, 1)
        init.xavier_normal_(self.init_mlp_z.weight, gain=1.0)
        init.xavier_normal_(self.init_mlp_r.weight, gain=1.0)
        init.xavier_normal_(self.init_mlp_hwave.weight, gain=1.0)

        ############ The layers for Iteration ############
        ### 1. Feature extraction
        self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        ### 2. Transformer
        self.self_attn = Transformer_self_attn(args=args)
        self.cross_attn = Transformer_cross_attn(args=args)
        ### 3. point-wise weight
        mlp_w, mlp_w_2 = [512, 256, 256, 256], [64]
        self.weight_mlp = MLPNet(self.state_dim, mlp_w, b_shared=True).layers
        self.weight_mlp_2 = MLPNet(256, mlp_w_2, b_shared=True).layers

        self.weight = torch.nn.Conv1d(64, 2, 1)
        init.xavier_normal_(self.weight.weight, gain=1.0)
        ### 4. Rotary center
        mlp_or, mlp_or_2, mlp_or_3 = [512, 256], [256, 128, 64], [64, 64, 64]
        self.or_mlp = MLPNet(self.state_dim, mlp_or, b_shared=True).layers
        self.or_mlp_2 = MLPNet(512, mlp_or_2, b_shared=True).layers
        self.or_mlp_3 = MLPNet(64, mlp_or_3, b_shared=True).layers

        self.or_ = torch.nn.Conv1d(64, 2, 1)
        init.xavier_normal_(self.or_.weight, gain=1.0)
        ### The layers of GRU
        ### 1. Acquire the Z; 2. Acquire the R; 3. Acquire the H-wave
        self.mlp_z = torch.nn.Conv1d(self.emb_dims + (args.score_top_k + self.emb_dims) * (args.view_num-1) + self.state_dim, self.state_dim, 1)
        self.mlp_r = torch.nn.Conv1d(self.emb_dims + (args.score_top_k + self.emb_dims) * (args.view_num-1) + self.state_dim, self.state_dim, 1)
        self.mlp_hwave = torch.nn.Conv1d(self.emb_dims + (args.score_top_k + self.emb_dims) * (args.view_num-1) + self.state_dim, self.state_dim, 1)
        init.xavier_normal_(self.mlp_z.weight, gain=1.0)
        init.xavier_normal_(self.mlp_r.weight, gain=1.0)
        init.xavier_normal_(self.mlp_hwave.weight, gain=1.0)

        ###### Others
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, Viewpoints, R, T, view_num, iteration=args.iteration):
        this_batch_size = Viewpoints.size()[0] # batch*view_num*point_num*3

        weight_list = []
        rotary_center_list = []
        modified_rotary_center_list = []
        align_viewpoints_list = []

        Viewpoints_list = []
        R_list = []
        T_list = []
        for i in range(args.view_num):
            if args.noise_type=='clean':
                Viewpoints_list.append(Viewpoints[:, i, :, :].transpose(1, 2)) # view_num batch*3*point_num
            elif args.noise_type=='jitter':
                Viewpoints_list.append(RandomJitter(Viewpoints[:, i, :, :].transpose(1, 2), scale=0.01, clip=0.05))  # view_num batch*3*point_num
            else:
                raise NotImplementedError

            R_list.append(R[:, i, :, :])  # view_num batch*3*3
            T_list.append(T[:, i, :])  # view_num batch*3

        # init the hidden state
        # batch*3*point_num-->batch*dim*point_num
        init_state_h = torch.tanh(self.init_emb_nn_state(Viewpoints_list[0], if_relu_atlast=False))  # h0 batch*dim*point_num

        for iter_stage in range(iteration):
            if iter_stage==0:
                views_embedding = []
                for i in range(view_num):
                    views_embedding.append(self.init_self_attn(self.init_emb_nn(Viewpoints_list[i])))  # fsa batch*dim*point_num

                view_ba = []
                for i in range(1, view_num):
                    b2a = self.init_cross_attn(views_embedding[i], views_embedding[i-1]) # fca batch*dim*point_num
                    view_ba.append(b2a)
                phi_view_ba = []
                for i in range(1, view_num):
                    phi_view_ba.append(view_ba[i-1] + views_embedding[i]) # batch*dim*point_num

                global_feat = []
                for i in range(1, view_num):
                    global_feat.append(symfn_avg(phi_view_ba[i - 1]).expand(-1, -1, phi_view_ba[i - 1].size()[-1]))  # batch*dim*point_num

                global_feat = torch.cat(global_feat, dim=1)

                scores_list = []
                for i in range(1, view_num):
                    d_k = views_embedding[i].size(1)
                    scores = torch.matmul(views_embedding[i].transpose(2, 1).contiguous(), views_embedding[i-1]) / math.sqrt(d_k)
                    scores_global, _ = scores.topk(k=args.score_top_k, dim=2)
                    scores_list.append(scores_global.transpose(1, 2).view(this_batch_size, args.score_top_k, -1))

                scores = torch.cat(scores_list, dim=1)

                # Let's view the feature_concate as Xk.
                feature_concat = torch.cat((global_feat, scores, views_embedding[0]), 1)  # xk
                # The GRU
                hx = torch.cat([feature_concat, init_state_h], dim=1)  # [xk, hk-1]
                new_z = self.sigmoid(self.init_mlp_z(hx))
                new_r = self.sigmoid(self.init_mlp_r(hx))
                new_hwave = torch.tanh(self.init_mlp_hwave(torch.cat([torch.mul(new_r, init_state_h), feature_concat], dim=1)))
                state_h = torch.mul(new_hwave, new_z) + torch.mul(init_state_h, (1 - new_z))  # hk batch*dim*point_num

                weight_feature = self.init_weight_mlp(state_h) # batch*dim*point_num
                phi_feature = self.init_or_mlp(state_h) # batch*dim*point_num
                phi_feature = torch.cat([weight_feature, phi_feature], dim=1)
                phi_feature = self.init_or_mlp_2(phi_feature) # batch*dim*point_num
                phi_feature_pooling = symfn_avg(phi_feature).view(this_batch_size, -1, 1) # batch*dim*1
                phi_feature_pooling = self.init_or_mlp_3(phi_feature_pooling)
                OR = self.init_or_(phi_feature_pooling).view(this_batch_size, 2)  # phi_k batch*2
                OR = torch.cat([OR, torch.zeros(this_batch_size, 1).cuda()], dim=-1) # batch*3
                modified_OR = OR

                # batch*3, view_num batch*3*point_num, view_num batch*3*3, view_num batch*3---> view_num batch*3*point_num
                align_viewpoints = Alignment(modified_OR, Viewpoints_list, R_list, T_list, view_num)

                # OR_init, modified_OR, align_viewpoints: batch*3, batch*3, view_num batch*3*point_num
                OR_init, modified_OR, align_viewpoints, state_h = OR, modified_OR, align_viewpoints, state_h

                weight_list.append(torch.cat([torch.ones(this_batch_size, 2), torch.zeros(this_batch_size, 1)], dim=-1).cuda())  # batch*3
                rotary_center_list.append(OR_init)  # batch*3
                modified_rotary_center_list.append(modified_OR)  # batch*3
                align_viewpoints_list.append(align_viewpoints)  # view_num batch*3*point_num

            else:
                align_views = []
                for i in range(view_num):
                    align_views.append(align_viewpoints_list[iter_stage-1][i].detach()) # batch*3*point_num

                views_embedding = []
                for i in range(view_num):
                    views_embedding.append(self.self_attn(self.emb_nn(align_views[i])))  # fsa batch*dim*point_num

                view_ba = []
                for i in range(1, view_num):
                    b2a = self.cross_attn(views_embedding[i], views_embedding[i - 1]) # fca batch*dim*point_num
                    view_ba.append(b2a)
                phi_view_ba = []
                for i in range(1, view_num):
                    phi_view_ba.append(view_ba[i - 1] + views_embedding[i])  # batch*dim*point_num

                global_feat = []
                for i in range(1, view_num):
                    global_feat.append(symfn_avg(phi_view_ba[i - 1]).expand(-1, -1, phi_view_ba[i - 1].size()[-1]))  # batch*dim*point_num

                global_feat = torch.cat(global_feat, dim=1)

                scores_list = []
                for i in range(1, view_num):
                    d_k = views_embedding[i].size(1)
                    scores = torch.matmul(views_embedding[i].transpose(2, 1).contiguous(), views_embedding[i-1]) / math.sqrt(d_k)
                    scores_global, _ = scores.topk(k=args.score_top_k, dim=2)
                    scores_list.append(scores_global.transpose(1, 2).view(this_batch_size, args.score_top_k, -1))

                scores = torch.cat(scores_list, dim=1)

                # Let's view the feature_concate as Xt.
                feature_concat = torch.cat((global_feat, scores, views_embedding[0]), 1)  # xk
                # The GRU
                hx = torch.cat([feature_concat, state_h], dim=1)  # [xk, hk-1]
                new_z = self.sigmoid(self.mlp_z(hx))
                new_r = self.sigmoid(self.mlp_r(hx))
                new_hwave = torch.tanh(self.mlp_hwave(torch.cat([torch.mul(new_r, state_h), feature_concat], dim=1)))
                state_h = torch.mul(new_hwave, new_z) + torch.mul(state_h, (1 - new_z))   # hk batch*dim*point_num

                weight_feature = self.weight_mlp(state_h) # batch*dim*point_num
                phi_feature = self.or_mlp(state_h) # batch*dim*point_num
                phi_feature = torch.cat([weight_feature, phi_feature], dim=1)

                weight_feature = self.weight_mlp_2(weight_feature)
                weight_feature = symfn_avg(weight_feature).view(this_batch_size, -1, 1) # batch*dim*1
                weight = self.weight(weight_feature).view(this_batch_size, 2)  # wk batch*2
                weight = torch.cat([weight,  torch.zeros(this_batch_size, 1).cuda()], dim=-1) # batch*3

                phi_feature = self.or_mlp_2(phi_feature) # batch*dim*point_num
                phi_feature_pooling = symfn_avg(phi_feature).view(this_batch_size, -1, 1) # batch*dim*1
                phi_feature_pooling = self.or_mlp_3(phi_feature_pooling)
                OR = self.or_(phi_feature_pooling).view(this_batch_size, 2)  # phi_k batch*2
                OR = torch.cat([OR, torch.zeros(this_batch_size, 1).cuda()], dim=-1) # batch*3
                modified_OR = torch.mul(weight, OR) + torch.mul((1 - weight), modified_rotary_center_list[iter_stage-1])  # 等式(6) batch*3

                # batch*3, view_num batch*3*point_num, view_num batch*3*3, view_num batch*3---> view_num batch*3*point_num
                align_viewpoints = Alignment(modified_OR, Viewpoints_list, R_list, T_list, view_num)

                weight_list.append(weight) # batch*3
                rotary_center_list.append(OR) # batch*3
                modified_rotary_center_list.append(modified_OR) # batch*3
                align_viewpoints_list.append(align_viewpoints) # view_num batch*3*point_num

        return weight_list, rotary_center_list, modified_rotary_center_list, align_viewpoints_list
