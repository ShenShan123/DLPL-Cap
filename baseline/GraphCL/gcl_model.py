import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class graphcl(nn.Module):
    """
    SRAM graph contrastive learning model for subgraph-level representation
    """
    def __init__(self, gnn):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        
        # 获取GNN的输出维度
        if hasattr(gnn, 'out_channels'):
            out_dim = gnn.out_channels
        else:
            # 默认维度，适用于大多数情况
            out_dim = 128
            
        self.projection_head = nn.Sequential(
            nn.Linear(out_dim, out_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(out_dim, out_dim)
        )

    def forward_cl(self, x, edge_index, edge_attr, batch):
        """
        Calculate the subgraph representation
        """
        # 确保x是浮点类型
        if x.dtype != torch.float32 and x.dtype != torch.float64:
            x = x.float()
            
        # 检查是否可以传递edge_attr参数
        if edge_attr is not None:
            try:
                x = self.gnn(x, edge_index, edge_attr)
            except TypeError:
                # 如果传递edge_attr参数出错，则不传递
                x = self.gnn(x, edge_index)
        else:
            x = self.gnn(x, edge_index)
        
        # 池化得到子图级表示
        x = self.pool(x, batch)
        
        # 投影头
        x = self.projection_head(x)
        return x
    
    def encode_subgraph(self, subgraph):
        """
        Encode a single subgraph data object
        """
        x, edge_index = subgraph.x, subgraph.edge_index
        edge_attr = subgraph.edge_type if hasattr(subgraph, 'edge_type') else None
        batch = subgraph.batch if hasattr(subgraph, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        return self.forward_cl(x, edge_index, edge_attr, batch)

    def loss_cl(self, x1, x2):
        """
        Calculate the contrastive loss between two sets of subgraph representations
        """
        T = 0.1  # Temperature parameter
        batch_size, _ = x1.size()
        
        # Normalize features for cosine similarity
        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)
        
        # Calculate similarity matrix
        sim_matrix = torch.matmul(x1, x2.T) / T
        
        # Positive pairs are on the diagonal
        pos_sim = torch.diag(sim_matrix)
        
        # For each sample, compute InfoNCE loss
        neg_sim = torch.logsumexp(sim_matrix, dim=1)
        loss = -torch.mean(pos_sim - neg_sim)
        
        return loss