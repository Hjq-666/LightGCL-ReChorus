import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as torch_fun

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
    def build_adjmat(user_count, item_count, train_mat, selfloop_flag=False):
        """
        构建归一化的邻接矩阵
        """
        interaction_matrix = sp.dok_matrix((user_count, item_count), dtype=np.float32)
        
        # 填充用户-物品交互矩阵
        for user, items in train_mat.items():
            for item in items:
                interaction_matrix[user, item] = 1.0
        
        interaction_matrix = interaction_matrix.tocoo()
        row_sum = np.array(interaction_matrix.sum(1)).flatten() + 1e-10
        col_sum = np.array(interaction_matrix.sum(0)).flatten() + 1e-10
        
        # 归一化处理
        row_indices, col_indices = interaction_matrix.row, interaction_matrix.col
        data = interaction_matrix.data / np.sqrt(row_sum[row_indices] * col_sum[col_indices])
        
        normalized_adj_matrix = sp.csr_matrix((data, (row_indices, col_indices)), shape=(user_count, item_count))
        return normalized_adj_matrix

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
        bpr_loss = -torch.log(torch.sigmoid(pos_pred[:, None] - neg_pred) + 1e-10).mean()
        return bpr_loss

    def _compute_cl_loss(self, out_dict):
        """
        计算对比学习损失
        """
        g_user_vector, user_vector, g_item_vector, item_vector = out_dict['g_user_vector'], out_dict['user_vector'], out_dict['g_item_vector'], out_dict['item_vector']
        g_user_norm = torch_fun.normalize(g_user_vector, p=2, dim=-1)
        user_norm = torch_fun.normalize(user_vector, p=2, dim=-1)
        g_item_norm = torch_fun.normalize(g_item_vector, p=2, dim=-1)
        item_norm = torch_fun.normalize(item_vector, p=2, dim=-1)
        
        # 计算正样本得分
        pos_score = torch.cat([(g_user_norm * user_norm).sum(dim=-1), (g_item_norm * item_norm).sum(dim=-1)], dim=0) / self.temperature
        
        # 计算负样本得分
        num_items = g_user_norm.size(1)
        mask = torch.eye(num_items, dtype=torch.bool).unsqueeze(0).to(g_user_norm.device)
        neg_score_user = (g_user_norm @ user_norm.transpose(1, 2) / self.temperature).masked_fill(mask, float('-inf'))
        neg_score_item = (g_item_norm @ item_norm.transpose(1, 2) / self.temperature).masked_fill(mask, float('-inf'))
        neg_score = torch.cat([neg_score_user, neg_score_item], dim=0)
        
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
    def calculate_svd(norm_adj, q):
        """
        计算SVD分解
        """
        svd_u, svd_v, svd_d = torch.svd_lowrank(norm_adj, q=q)
        return svd_u, svd_v, svd_d

    def forward(self, users, items):
        """
        前向传播
        """
        # 执行SVD分解
        svd_u, svd_s, svd_v = self.calculate_svd(self.sparse_norm_adj, q=self.rank)
        u_mul_s = torch.matmul(svd_u, torch.diag(svd_s))
        v_mul_s = torch.matmul(svd_v, torch.diag(svd_s))
        vt = svd_v.T
        ut = svd_u.T

        # 初始化用户和物品嵌入
        user_embedding_list, item_embedding_list = [self.embedding_dict['user_emb']], [self.embedding_dict['item_emb']]
        g_user_embedding_list, g_item_embedding_list = [], []

        # 多层传播
        for layer in range(self.num_layers):
            # GNN传播
            Z_u = torch.spmm(self.sparse_norm_adj, item_embedding_list[layer])
            Z_i = torch.spmm(self.sparse_norm_adj.T, user_embedding_list[layer])

            # SVD传播
            G_u = torch.matmul(u_mul_s, torch.matmul(vt, item_embedding_list[layer]))
            G_i = torch.matmul(v_mul_s, torch.matmul(ut, user_embedding_list[layer]))

            # 聚合结果
            user_embedding_list.append(Z_u)
            item_embedding_list.append(Z_i)
            g_user_embedding_list.append(G_u)
            g_item_embedding_list.append(G_i)

        # 汇总所有层的结果
        G_u = sum(g_user_embedding_list)
        G_i = sum(g_item_embedding_list)
        E_u = sum(user_embedding_list)
        E_i = sum(item_embedding_list)

        # 获取所有用户和物品的嵌入
        user_all_embeddings = E_u
        item_all_embeddings = E_i

        # 提取指定用户和物品的嵌入
        user_embeddings = user_all_embeddings[users]
        item_embeddings = item_all_embeddings[items]
        G_user_embeddings = G_u[users]
        G_item_embeddings = G_i[items]

        # 返回原始和传播后的嵌入
        return user_embeddings, item_embeddings, G_user_embeddings, G_item_embeddings
