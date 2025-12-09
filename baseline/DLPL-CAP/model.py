import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, ClusterGCNConv
from torch_geometric.nn.models.mlp import MLP

NET = 0
DEV = 1
PIN = 2

class CapClassifier(nn.Module):
    """GNN-based classifier for capacitance classification"""
    def __init__(self, hidden_dim, num_classes=None, class_boundaries=None, 
                 num_layers=2, use_bn=False, drop_out=0.0, 
                 activation='relu', src_dst_agg='concat', gnn_type='sage', task_type='regression', use_stats=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.src_dst_agg = src_dst_agg
        self.task_type = task_type
        # Set default class boundaries if not provided
        self.class_boundaries = class_boundaries or [0.33,0.67]
        if num_classes != len(self.class_boundaries) + 1:
            print(f"Adjusting num_classes to {len(self.class_boundaries)+1}")
            self.num_classes = len(self.class_boundaries) + 1
        
        self.use_stats = use_stats
        ## Circuit Statistics encoder
        if self.use_stats:
            node_embed_dim = int(hidden_dim / 2)
            c_embed_dim = hidden_dim % node_embed_dim + node_embed_dim
            self.c_embed_dim = c_embed_dim
            # add node_attr transform layer for net/device/pin nodes, by shan
            self.net_attr_layers = nn.Linear(17, c_embed_dim, bias=True)
            self.dev_attr_layers = nn.Linear(17, c_embed_dim, bias=True)
            self.pin_attr_layers = nn.Embedding(17, c_embed_dim)
        else:
            node_embed_dim = hidden_dim
        # Node encoders
        self.node_encoder = nn.Embedding(num_embeddings=4, embedding_dim=node_embed_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == 'sage':
                self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim, aggr='mean'))
            elif gnn_type == 'gcn':
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim, improved=True))
            elif gnn_type == 'gat':
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=1, concat=False))
            else:  # default is cluster
                self.gnn_layers.append(ClusterGCNConv(hidden_dim, hidden_dim, improved=True))
        
        # For graph pooling if needed
        if src_dst_agg == 'pool':
            self.pooling_fun = pygnn.pool.global_mean_pool
        
        # Classification MLP
        # we need to predict the class for each edge to perform the edge regression task
        head_input_dim = hidden_dim * 2 if src_dst_agg == 'concat' else hidden_dim
        if self.task_type == 'regression':
            self.classifier = MLP(
                in_channels=head_input_dim,
                hidden_channels=hidden_dim,
                out_channels=self.num_classes,
                num_layers=3,
                dropout=drop_out,
                norm='batch_norm' if use_bn else None,
                activation=activation
            )
        else:
            # classification   we need to predict the capacitance value for each edge to predict whether the edge is a coupling edge or not
            self.classifier = MLP(      
                in_channels=head_input_dim,
                hidden_channels=hidden_dim,
                out_channels=1,
                num_layers=3,
                dropout=drop_out,
                norm='batch_norm' if use_bn else None,
                activation=activation
            )
        
        self.use_bn = use_bn
        self.drop_out = drop_out
        self.bn_node_x = nn.BatchNorm1d(hidden_dim) if use_bn else None
        self.activation = F.relu if activation == 'relu' else F.elu
    
    def classify_capacitance(self, capacitance_value):
        if self.task_type == 'regression':
            """Classify capacitance values into appropriate class"""
            if isinstance(capacitance_value, torch.Tensor):
                classes = torch.zeros_like(capacitance_value, dtype=torch.long)
                for i, boundary in enumerate(self.class_boundaries):
                    classes = torch.where(capacitance_value < boundary, i, classes)
                return classes
            else:
                for i, boundary in enumerate(self.class_boundaries):
                    if capacitance_value < boundary:
                        return i
                return len(self.class_boundaries)
        else:
            return capacitance_value
    
    def forward(self, batch):
        """Forward pass through the classifier"""
        # Node encoding
        z = self.node_encoder(batch.x[:, 0])

        # use circuit statistics encoder
        ## If we use circuit statistics encoder
        if self.use_stats:
            node_attr_emb = torch.zeros((batch.num_nodes, self.c_embed_dim), device=batch.x.device)
            
            # 获取节点类型
            net_node_mask = batch.x.squeeze() == NET
            dev_node_mask = batch.x.squeeze() == DEV
            pin_node_mask = batch.x.squeeze() == PIN
            
            # 使用node_attr作为节点特征
           # 在node_attr_emb赋值前，确保类型匹配
            node_attr_emb[net_node_mask] = self.net_attr_layers(batch.node_attr[net_node_mask]).to(node_attr_emb.dtype)
            node_attr_emb[dev_node_mask] = self.dev_attr_layers(batch.node_attr[dev_node_mask]).to(node_attr_emb.dtype)
            node_attr_emb[pin_node_mask] = self.pin_attr_layers(batch.node_attr[pin_node_mask, 0].long()).to(node_attr_emb.dtype)
            ## concatenate node embeddings and circuit statistics embeddings
            z = torch.cat((z, node_attr_emb), dim=1)
        
        # Apply GNN layers
        for conv in self.gnn_layers:
            # 确保数据类型和设备正确
            if z.device != batch.edge_index.device:
                batch.edge_index = batch.edge_index.to(z.device)
                
            if z.dtype != torch.float32:
                z = z.float()
                
            if batch.edge_index.dtype != torch.long:
                batch.edge_index = batch.edge_index.long()
                
            # 检查边索引是否有效
            if torch.any(batch.edge_index >= z.size(0)):
                # 过滤掉无效的边
                valid_edges = (batch.edge_index[0] < z.size(0)) & (batch.edge_index[1] < z.size(0))
                edge_index = batch.edge_index[:, valid_edges]
                # 打印警告，但继续处理
                print(f"Warning: Filtered {(~valid_edges).sum().item()} invalid edges")
            else:
                edge_index = batch.edge_index
                
            # 直接应用图卷积，让PyTorch Geometric处理批处理
            z = conv(z, edge_index)
                
            if self.bn_node_x is not None:
                z = self.bn_node_x(z)
            z = self.activation(z)
            if self.drop_out > 0.0:
                z = F.dropout(z, p=self.drop_out, training=self.training)

        # Aggregate node embeddings based on strategy
        if self.src_dst_agg == 'pool':
            graph_emb = self.pooling_fun(z, batch.batch)
        else:
            batch_size = batch.edge_label.size(0)
            src_emb = z[:batch_size, :]
            dst_emb = z[batch_size:batch_size*2, :]
            
            if self.src_dst_agg == 'concat':
                graph_emb = torch.cat((src_emb, dst_emb), dim=1)
            else:  # 'add'
                graph_emb = src_emb + dst_emb
        
        # Classification
        return self.classifier(graph_emb),graph_emb


class CapRegressor(nn.Module):
    """Class-specific capacitance regression model (second stage)"""
    def __init__(self, hidden_dim, num_classes=3, 
                 use_bn=False, drop_out=0.0, activation='relu', src_dst_agg='concat'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Create regression heads for each class (4-Layer MLP for each class)
        self.regressors = nn.ModuleList()
        head_input_dim = hidden_dim * 2 if src_dst_agg == 'concat' else hidden_dim
        for i in range(self.num_classes):
            
            self.regressors.append(MLP(
                in_channels=head_input_dim,  # Takes network embedding as input
                hidden_channels=int(hidden_dim/num_classes),  # Convert to integer
                out_channels=1,  # Output is capacitance value
                num_layers=4,    # 4-Layer MLP 
                dropout=drop_out,
                norm='batch_norm' if use_bn else None,
                activation=activation
            ))
    
    def forward(self, graph_emb, class_idx):
        """Forward pass handling batches with different classes (vectorized)
        
        Args:
            graph_emb: graph embedding from GNN
                Shape: [batch_size, embedding_dim]
            class_idx: Class indices for each sample in the batch
                Shape: [batch_size]
        """
        batch_size = graph_emb.shape[0]
        
        # Validate all class indices
        if torch.any((class_idx < 0) | (class_idx >= self.num_classes)):
            invalid_indices = class_idx[(class_idx < 0) | (class_idx >= self.num_classes)]
            raise ValueError(f"Invalid class indices: {invalid_indices}")
        
        # Initialize output tensor with zeros
        # Determine output dimension - use a safer approach that doesn't cause BatchNorm issues
        output_shape = 1  # Default output dimension
        
        # Process each class separately and fill in the corresponding outputs
        outputs = None
        for c in range(self.num_classes):
            # Create a mask for samples of this class
            mask = (class_idx == c)
            
            # Check if we have any samples of this class
            if not torch.any(mask):
                continue
                
            # Get embeddings for this class
            class_emb = graph_emb[mask]
            
            # Forward through the regressor for this class
            class_output = self.regressors[c](class_emb)
            
            # Initialize the outputs tensor if not done yet
            if outputs is None:
                outputs = torch.zeros((batch_size, output_shape), device=graph_emb.device)
            
            # Fill in the outputs for this class
            # 在outputs[mask] = class_output前，确保类型匹配
            class_output = class_output.to(outputs.dtype)
            outputs[mask] = class_output
        
        # In case there were no valid predictions, output all zeros
        if outputs is None:
            outputs = torch.zeros((batch_size, output_shape), device=graph_emb.device)
            
        return outputs.squeeze(-1)

    def predict_capacitance(self, graph_emb, classes):
        """
        预测不同类别下的电容值
        
        参数:
            graph_emb: 图嵌入特征，形状为 [batch_size, embedding_dim]
            classes: 样本类别索引，形状为 [batch_size]
            
        返回:
            预测的电容值，形状为 [batch_size]
        """
        # 确保classes是长整型
        classes = classes.long()
        
        # 使用forward方法进行预测
        return self.forward(graph_emb, classes)