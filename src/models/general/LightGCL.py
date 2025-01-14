import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

from models.BaseModel import GeneralModel

class LightGCL(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'n_layers', 'batch_size', 'q', 'temp', 'lambda1', 'lambda2', 'hyper_layers']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64, help='Size of embedding vectors.')
        parser.add_argument('--n_layers', type=int, default=2, help='Number of LightGCL layers.')
        parser.add_argument('--q', type=int, default=5, help='rank')
        parser.add_argument('--temp', default=0.2, type=float, help='temperature in cl loss')
        parser.add_argument('--lambda1', default=0.2, type=float, help='weight of cl loss')
        parser.add_argument('--lambda2', default=1e-7, type=float, help='l2 reg weight')
        parser.add_argument('--hyper_layers', default=1, type=int, help='number of hyper layers')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        GeneralModel.__init__(self, args, corpus)
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.norm_adj = self.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set)
        self.q = args.q
        self.temp = args.temp
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.encoder = LGCLEncoder(corpus.n_users, corpus.n_items, self.emb_size, self.norm_adj, self.n_layers, self.q, self.temp, self.lambda1, self.lambda2)
        self.apply(self.init_weights)

    @staticmethod
    def normalized_adj_single(adj):
        """
        计算归一化后的邻接矩阵
        """
        rowsum = np.array(adj.sum(1)) + 1e-10
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    @staticmethod
    def build_adjmat(user_count, item_count, train_mat, selfloop_flag=False):
        """
        构建归一化的邻接矩阵
        """
        R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
        
        # 填充用户-物品交互矩阵
        for user in train_mat:
            for item in train_mat[user]:
                R[user, item] = 1.0
        
        R = R.tocoo()
        rowD = np.array(R.sum(1)).flatten() + 1e-10
        colD = np.array(R.sum(0)).flatten() + 1e-10
        R_coo = R.tocoo()
        data = R_coo.data
        row = R_coo.row
        col = R_coo.col
        
        # 归一化处理
        for i in range(len(data)):
            data[i] = data[i] / pow(rowD[row[i]] * colD[col[i]], 0.5)
        
        norm_adj_mat = sp.csr_matrix((data, (row, col)), shape=(user_count, item_count))
        return norm_adj_mat

    def forward(self, feed_dict):
        """
        前向传播
        """
        user, items = feed_dict['user_id'], feed_dict['item_id']
        user_embed, item_embed, g_user_embed, g_item_embed = self.encoder(user, items)
        
        # 计算预测评分
        prediction = (user_embed[:, None, :] * item_embed).sum(dim=-1)
        
        # 计算用户和物品的嵌入
        u_v = user_embed.repeat(1, items.shape[1]).view(items.shape[0], items.shape[1], -1)
        g_u_v = g_user_embed.repeat(1, items.shape[1]).view(items.shape[0], items.shape[1], -1)
        i_v = item_embed
        g_i_v = g_item_embed
        
        return {
            'prediction': prediction.view(feed_dict['batch_size'], -1),
            'u_v': u_v,
            'i_v': i_v,
            'g_u_v': g_u_v,
            'g_i_v': g_i_v
        }

    def loss(self, out_dict):
        """
        计算损失
        """
        prediction = out_dict['prediction']
        bpr_loss = self._compute_bpr_loss(prediction)
        cl_loss = self._compute_cl_loss(out_dict)
        reg_loss = self._compute_reg_loss()
        loss = bpr_loss + self.lambda1 * cl_loss + self.lambda2 * reg_loss
        return loss

    def _compute_bpr_loss(self, prediction):
        """
        计算BPR损失
        """
        pos_pred, neg_pred = prediction[:, 0], prediction[:, 1:]
        neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
        bpr_loss = -(((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1)).clamp(min=1e-8, max=1-1e-8).log().mean()
        return bpr_loss

    def _compute_cl_loss(self, out_dict):
        """
        计算对比学习损失
        """
        G_u, E_u, G_i, E_i = out_dict['g_u_v'], out_dict['u_v'], out_dict['g_i_v'], out_dict['i_v']
        G_u_norm = F.normalize(G_u, p=2, dim=-1)
        E_u_norm = F.normalize(E_u, p=2, dim=-1)
        G_i_norm = F.normalize(G_i, p=2, dim=-1)
        E_i_norm = F.normalize(E_i, p=2, dim=-1)
        
        # 计算正样本得分
        pos_score_u = (G_u_norm * E_u_norm).sum(dim=-1) / self.temp
        pos_score_i = (G_i_norm * E_i_norm).sum(dim=-1) / self.temp
        pos_score = torch.cat([pos_score_u, pos_score_i], dim=0)
        
        # 计算负样本得分
        batch_size, num_items = G_u_norm.size(0), G_u_norm.size(1)
        mask = torch.eye(num_items, dtype=torch.bool).unsqueeze(0).to(G_u_norm.device)
        neg_score_u = torch.matmul(G_u_norm, E_u_norm.transpose(1, 2)) / self.temp
        neg_score_u = neg_score_u.masked_fill(mask, float('-inf'))
        neg_score_i = torch.matmul(G_i_norm, E_i_norm.transpose(1, 2)) / self.temp
        neg_score_i = neg_score_i.masked_fill(mask, float('-inf'))
        neg_score = torch.cat([neg_score_u, neg_score_i], dim=0)
        
        # 计算对比学习损失
        neg_logsumexp = torch.logsumexp(neg_score, dim=-1)
        cl_loss = - (pos_score - neg_logsumexp).mean()
        return cl_loss

    def _compute_reg_loss(self):
        """
        计算正则化损失
        """
        reg_loss = 0
        for param in self.parameters():
            reg_loss += param.norm(2).square()
        return reg_loss

class LGCLEncoder(nn.Module):
    def __init__(self, user_count, item_count, emb_size, norm_adj, n_layers, q, temp, lambda1, lambda2):
        super(LGCLEncoder, self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.emb_size = emb_size
        self.norm_adj = norm_adj
        self.n_layers = n_layers
        self.q = q
        self.temp = temp
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).cuda()

    def _init_model(self):
        """
        初始化模型参数
        """
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.emb_size))),
        })
        return embedding_dict

    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
        """
        将稀疏矩阵转换为稀疏张量
        """
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    @staticmethod
    def cal_svd_s_v_d(norm_adj, q):
        """
        计算SVD分解
        """
        svd_u, svd_v, svd_d = torch.svd_lowrank(norm_adj, q=q)
        return svd_u, svd_v, svd_d

    def forward(self, users, items):
        """
        前向传播
        """
        # 计算SVD分解
        svd_u, svd_s, svd_v = self.cal_svd_s_v_d(self.sparse_norm_adj, q=self.q)
        u_mul_s = svd_u @ torch.diag(svd_s)
        v_mul_s = svd_v @ torch.diag(svd_s)
        vt = svd_v.T
        ut = svd_u.T

        # 初始化用户和物品的嵌入
        E_u_list = [self.embedding_dict['user_emb']]
        E_i_list = [self.embedding_dict['item_emb']]
        G_u_list = []
        G_i_list = []

        # 进行多层传播
        for layer in range(1, self.n_layers + 1):
            # GNN传播
            Z_u = torch.spmm(self.sparse_norm_adj, E_i_list[layer - 1])
            Z_i = torch.spmm(self.sparse_norm_adj.T, E_u_list[layer - 1])

            # SVD传播
            vt_ei = vt @ E_i_list[layer - 1]
            G_u = u_mul_s @ vt_ei
            ut_eu = ut @ E_u_list[layer - 1]
            G_i = v_mul_s @ ut_eu

            # 聚合结果
            E_u_list.append(Z_u)
            E_i_list.append(Z_i)
            G_u_list.append(G_u)
            G_i_list.append(G_i)

        # 对所有层的结果求和
        G_u = sum(G_u_list)
        G_i = sum(G_i_list)
        E_u = sum(E_u_list)
        E_i = sum(E_i_list)

        # 获取所有用户和物品的嵌入
        user_all_embeddings = E_u
        item_all_embeddings = E_i

        # 获取指定用户和物品的嵌入
        user_embeddings = user_all_embeddings[users, :]
        item_embeddings = item_all_embeddings[items, :]
        G_user_embeddings = G_u[users, :]
        G_item_embeddings = G_i[items, :]

        # 返回原视图的嵌入和新构建图的嵌入
        return user_embeddings, item_embeddings, G_user_embeddings, G_item_embeddings