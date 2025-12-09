import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, ResGatedGraphConv, GINConv, ChebConv, GINEConv, ClusterGCNConv, SSGConv
from torch_geometric.nn.models.mlp import MLP
from torch_geometric.data import Data

NET = 0
DEV = 1
PIN = 2

class GraphHead(nn.Module):
    """ GNN head for graph-level prediction.

    Implementation adapted from the transductive GraphGPS.

    Args:
        hidden_dim (int): Hidden features' dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        num_layers (int): Number of layers of GNN model
        layers_post_mp (int): number of layers of head MLP
        use_bn (bool): whether to use batch normalization
        drop_out (float): dropout rate
        activation (str): activation function
        src_dst_agg (str): the way to aggregate src and dst nodes, which can be 'concat' or 'add' or 'pool'
    """
    def __init__(self, hidden_dim, dim_out, num_layers=2, num_head_layers=2, 
                 use_bn=False, drop_out=0.0, activation='relu', 
                 src_dst_agg='concat',  max_dist=400, 
                 task='classification'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task = task
        
        # 减小节点嵌入维度
        node_embed_dim = min(hidden_dim, 64)  # 限制最大嵌入维度

        ## Node / Edge type encoders
        self.node_encoder = nn.Embedding(num_embeddings=4,
                                         embedding_dim=node_embed_dim)
        self.edge_encoder = nn.Embedding(num_embeddings=10,
                                         embedding_dim=node_embed_dim)
        
        # GNN layers - 使用更轻量级的配置
        self.layers = nn.ModuleList()
        self.drop_out = min(drop_out, 0.3)  # 限制最大dropout
        self.use_bn = use_bn
        for _ in range(num_layers):
            self.layers.append(SAGEConv(node_embed_dim, node_embed_dim, 
                                      normalize=True,  # 添加归一化
                                      bias=True))
        
        ## The head configuration
        head_input_dim = node_embed_dim
        self.src_dst_agg = src_dst_agg
        head_input_dim = node_embed_dim * 2 if src_dst_agg == 'concat' else node_embed_dim
        if src_dst_agg == 'pool':
            self.pooling_fun = pygnn.pool.global_mean_pool

        self.num_head_layers = num_head_layers

        # head MLP layers
        self.head_layers = MLP(
            in_channels=head_input_dim, 
            hidden_channels=node_embed_dim, 
            out_channels=dim_out, 
            num_layers=num_head_layers, 
            use_bn=False, dropout=0.0, activation=activation
        )
        self.bn_node_x = nn.BatchNorm1d(node_embed_dim)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Invalid activation')

    def forward(self, batch):
        try:
            # 添加调试信息
            #print(f"Number of nodes: {batch.x.shape[0]}")
            #print(f"Number of edges: {batch.edge_index.shape[1]}")
            #print(f"Edge label shape: {batch.edge_label.shape if hasattr(batch, 'edge_label') else 'No edge label'}")
            
            # 检查 CUDA 内存使用
            #if torch.cuda.is_available():
            #    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            #    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            
            # 原有的 forward 逻辑
            z = self.node_encoder(batch.x[:, 0])

            for conv in self.layers:
                # 确保边索引在正确的设备上
                edge_index = batch.edge_index.to(z.device)
                z = conv(z, edge_index)
                ## for models that also take edge_attr as input
                # z = conv(z, batch.edge_index, edge_attr=ze)

                if self.use_bn:
                    z = self.bn_node_x(z)
                z = self.activation(z)
                if self.drop_out > 0.0:
                    z = F.dropout(z, p=self.drop_out, training=self.training)

            ## In head layers. If we use graph pooling, we need to call the pooling function here
            if self.src_dst_agg == 'pool':
                graph_emb = self.pooling_fun(z, batch.batch)
            ## Otherwise, only 2 anchor nodes are used to final prediction.
            else:
                batch_size = batch.edge_label.size(0)
                ## In the LinkNeighbor loader, the first batch_size nodes in z are source nodes and,
                ## the second 'batch_size' nodes in z are destination nodes. 
                ## Remaining nodes are the neighbors.
                src_emb = z[:batch_size, :]
                dst_emb = z[batch_size:batch_size*2, :]
                if self.src_dst_agg == 'concat':
                    graph_emb = torch.cat((src_emb, dst_emb), dim=1)
                else:
                    graph_emb = src_emb + dst_emb

            pred = self.head_layers(graph_emb)

            return pred, batch.edge_label
            
        except RuntimeError as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Input shapes - x: {batch.x.shape}, edge_index: {batch.edge_index.shape}")
            if hasattr(batch, 'edge_label'):
                print(f"Edge label shape: {batch.edge_label.shape}")
            raise

    

