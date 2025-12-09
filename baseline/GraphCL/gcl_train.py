from gcl_dataset_aug import SramDataset_aug
from tqdm import tqdm
import torch
import argparse
import torch.optim as optim
from gcl_model import graphcl
from sram_dataset import performat_SramDataset
import os
import sys
sys.path.append('..')
from utils_model_checkpoint import check_model_exists, save_model, load_model, get_model_params_dict
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, ResGatedGraphConv, GINConv, ChebConv, GINEConv, ClusterGCNConv
import torch.nn as nn
import torch.nn.functional as F


class GNN(torch.nn.Module):
    """
    图神经网络模型，支持多种图卷积层
    """
    def __init__(self, input_dim, hidden_dim, output_dim, conv_type='ClusterGCNConv'):
        super(GNN, self).__init__()
        
        # 选择图卷积层类型
        if conv_type == 'GCNConv':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, output_dim)
        elif conv_type == 'SAGEConv':
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.conv3 = SAGEConv(hidden_dim, output_dim)
        elif conv_type == 'GATConv':
            self.conv1 = GATConv(input_dim, hidden_dim)
            self.conv2 = GATConv(hidden_dim, hidden_dim)
            self.conv3 = GATConv(hidden_dim, output_dim)
        elif conv_type == 'ResGatedGraphConv':
            self.conv1 = ResGatedGraphConv(input_dim, hidden_dim)
            self.conv2 = ResGatedGraphConv(hidden_dim, hidden_dim)
            self.conv3 = ResGatedGraphConv(hidden_dim, output_dim)
        elif conv_type == 'GINConv':
            nn1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            nn3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
            self.conv1 = GINConv(nn1)
            self.conv2 = GINConv(nn2)
            self.conv3 = GINConv(nn3)
        elif conv_type == 'ChebConv':
            self.conv1 = ChebConv(input_dim, hidden_dim, K=2)
            self.conv2 = ChebConv(hidden_dim, hidden_dim, K=2)
            self.conv3 = ChebConv(hidden_dim, output_dim, K=2)
        elif conv_type == 'GINEConv':
            nn1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            nn3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
            self.conv1 = GINEConv(nn1)
            self.conv2 = GINEConv(nn2)
            self.conv3 = GINEConv(nn3)
        else:  # 默认使用 ClusterGCNConv
            self.conv1 = ClusterGCNConv(input_dim, hidden_dim)
            self.conv2 = ClusterGCNConv(hidden_dim, hidden_dim)
            self.conv3 = ClusterGCNConv(hidden_dim, output_dim)
        
        self.conv_type = conv_type
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # 针对不同图卷积层类型处理不同的参数
        if self.conv_type in ['GINEConv']:
            # 这些卷积需要边特征
            x = F.relu(self.conv1(x, edge_index, edge_attr))
            x = F.relu(self.conv2(x, edge_index, edge_attr))
            x = self.conv3(x, edge_index, edge_attr)
        else:
            # 这些卷积不需要边特征
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = self.conv3(x, edge_index)
        
        return x

'''
def gcl_train(args, model, device, train_loader, optimizer, model_dir=None, save_every=20):
    """
    训练GraphCL模型
    :param args: 参数配置
    :param model: GraphCL模型
    :param device: 设备
    :param train_loader: 预处理好的训练数据加载器
    :param optimizer: 优化器
    :param model_dir: 模型保存目录，如果为None则不保存
    :param save_every: 每多少个epoch保存一次模型
    :return: 训练得到的子图表示
    """
    # 从参数中提取数据集名称
    dataset_name = args.train_dataset.split('+')[0] if '+' in args.train_dataset else args.train_dataset
    
    # 获取模型参数字典，用于唯一标识模型
    model_params = get_model_params_dict(args, exclude_keys=['num_workers', 'gpu', 'epochs', 'batch_size', 'seed'])
    
    try:
        # 检查是否存在已训练的模型
        model_exists, model_path = check_model_exists("GraphCL", dataset_name, model_params)
        if model_exists:
            # 尝试加载已有模型
            print(f"发现已有GraphCL模型: {model_path}")
            try:
                model.load_state_dict(torch.load(model_path))
                print(f"成功加载模型: {model_path}")
                print("正在为已加载的模型计算子图嵌入...")
                
                # 计算子图嵌入
                model.eval()  # 设置为评估模式
                subgraph_embeddings = {}
                edge_id_mapping = {}
                
                with torch.no_grad():
                    for step, batch in enumerate(tqdm(train_loader, desc="计算子图嵌入")):
                        batch = batch.to(device)
                        # 获取原始图的表示
                        x = model.forward_cl(batch.x, batch.edge_index,
                                          batch.edge_type if hasattr(batch, 'edge_type') else None,
                                          batch.batch if hasattr(batch, 'batch') else None)
                        
                        # 保存子图表示
                        if hasattr(batch, 'input_id'):
                            for i, input_id in enumerate(batch.input_id):
                                edge_id = input_id.item()
                                subgraph_embeddings[edge_id] = x[i].detach().cpu()
                                if edge_id not in edge_id_mapping:
                                    batch_size = batch.batch.max().item() + 1 if hasattr(batch, 'batch') and batch.batch is not None else len(batch.input_id)
                                    edge_id_mapping[edge_id] = i + step * batch_size
                
                print(f"完成子图嵌入计算，共 {len(subgraph_embeddings)} 个子图")
                return subgraph_embeddings, edge_id_mapping
                
            except Exception as e:
                print(f"加载模型失败，错误信息: {e}")
    except Exception as e:
        print(f"检查模型时发生错误: {e}")
    
    print("开始训练新模型...")
    model.train()
    total_epochs = getattr(args, 'cl_epochs', 100)
    best_loss = float('inf')
    best_embeddings = {}
    best_edge_mapping = {}
    
    print(f"开始图对比学习训练，共 {total_epochs} 个轮次")
    
    for epoch in range(1, total_epochs + 1):
        train_loss_accum = 0
        subgraph_embeddings = {}
        edge_id_mapping = {}
        
        # 对每个批次进行处理
        for step, batch in enumerate(tqdm(train_loader, desc=f"GraphCL Training Epoch {epoch}/{total_epochs}")):
            # 将批次复制两份用于不同的增强
            batch1 = batch.clone().to(device)
            batch2 = batch.clone().to(device)
            
            # 对batch1应用第一种增强
            if args.aug1 == 'dropN':
                batch1 = apply_drop_nodes(batch1, args.aug_ratio1, device)
            elif args.aug1 == 'permE':
                batch1 = apply_permute_edges(batch1, args.aug_ratio1)
            elif args.aug1 == 'maskN':
                batch1 = apply_mask_nodes(batch1, args.aug_ratio1)
            elif args.aug1 == 'subgraph':
                batch1 = apply_subgraph(batch1, args.aug_ratio1)
                
            # 对batch2应用第二种增强
            if args.aug2 == 'dropN':
                batch2 = apply_drop_nodes(batch2, args.aug_ratio2, device)
            elif args.aug2 == 'permE':
                batch2 = apply_permute_edges(batch2, args.aug_ratio2)
            elif args.aug2 == 'maskN':
                batch2 = apply_mask_nodes(batch2, args.aug_ratio2)
            elif args.aug2 == 'subgraph':
                batch2 = apply_subgraph(batch2, args.aug_ratio2)
                
            # 记录边ID映射
            if hasattr(batch, 'input_id'):
                for i, input_id in enumerate(batch.input_id):
                    edge_id = input_id.item()
                    if edge_id not in edge_id_mapping:
                        batch_size = batch.batch.max().item() + 1 if hasattr(batch, 'batch') and batch.batch is not None else len(batch.input_id)
                        edge_id_mapping[edge_id] = i + step * batch_size
            
            optimizer.zero_grad()
            
            # 获取两个增强视图的子图表示
            x1 = model.forward_cl(batch1.x, batch1.edge_index, 
                                batch1.edge_type if hasattr(batch1, 'edge_type') else None, 
                                batch1.batch if hasattr(batch1, 'batch') else None)
            x2 = model.forward_cl(batch2.x, batch2.edge_index, 
                                batch2.edge_type if hasattr(batch2, 'edge_type') else None, 
                                batch2.batch if hasattr(batch2, 'batch') else None)
            
            # 保存子图表示用于下游任务
            if hasattr(batch, 'input_id'):
                for i, input_id in enumerate(batch.input_id):
                    edge_id = input_id.item()
                    subgraph_embeddings[edge_id] = ((x1[i] + x2[i]) / 2).detach().cpu()
            
            # 计算对比损失
            loss = model.loss_cl(x1, x2)
            
            loss.backward()
            optimizer.step()
            
            train_loss_accum += float(loss.detach().cpu().item())
        
        # 计算当前epoch的平均损失
        avg_loss = train_loss_accum / (step + 1) if step >= 0 else 0
        print(f"GraphCL Epoch {epoch}/{total_epochs}, 平均损失: {avg_loss:.6f}")
        
        # 保存当前模型和训练指标
        metrics = {"loss": avg_loss}
        is_best = avg_loss < best_loss
        
        if model_dir is not None:
            save_model(
                model=model,
                model_name="GraphCL",
                dataset_name=dataset_name,
                epoch=epoch,
                params=model_params,
                is_best=is_best,
                metrics=metrics
            )
        
        # 保存最佳结果
        if is_best:
            best_loss = avg_loss
            best_embeddings = subgraph_embeddings
            best_edge_mapping = edge_id_mapping
            print(f"找到更好的模型，损失: {best_loss:.6f}")
    
    print(f"图对比学习训练完成，最佳损失: {best_loss:.6f}")
    # 返回训练得到的子图表示
    return best_embeddings, best_edge_mapping

# 添加批次级别的增强函数
def apply_drop_nodes(batch, drop_ratio, device):
    """在批次级别应用节点删除增强"""
    node_num = batch.x.size(0)
    drop_num = int(node_num * drop_ratio)
    
    # 随机选择要保留的节点
    idx_perm = torch.randperm(node_num, device=device)
    idx_drop = idx_perm[:drop_num]
    mask = torch.ones(node_num, dtype=torch.bool, device=device)
    mask[idx_drop] = 0
    
    # 更新节点特征
    batch.x = batch.x[mask]
    
    # 更新批次其他属性
    return update_batch_after_augmentation(batch, mask)

def apply_permute_edges(batch, perm_ratio):
    """在批次级别应用边扰动增强"""
    edge_num = batch.edge_index.size(1)
    perm_num = int(edge_num * perm_ratio)
    
    # 随机选择要保留的边
    idx_perm = torch.randperm(edge_num)
    idx_nondrop = idx_perm[perm_num:]
    
    # 更新边索引
    batch.edge_index = batch.edge_index[:, idx_nondrop]
    
    # 更新边属性（如果存在）
    if hasattr(batch, 'edge_type') and batch.edge_type is not None:
        batch.edge_type = batch.edge_type[idx_nondrop]
    if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
        batch.edge_attr = batch.edge_attr[idx_nondrop]
        
    return batch

def apply_mask_nodes(batch, mask_ratio):
    """在批次级别应用节点特征掩码增强"""
    node_num = batch.x.size(0)
    mask_num = int(node_num * mask_ratio)
    
    # 计算平均特征作为掩码值
    token = batch.x.mean(dim=0)
    
    # 随机选择要掩码的节点
    idx_mask = torch.randperm(node_num)[:mask_num]
    batch.x[idx_mask] = token
    
    return batch

def apply_subgraph(batch, ratio):
    """在批次级别应用子图提取增强"""
    # 子图提取比较复杂，暂时简化为节点删除的变体
    return apply_drop_nodes(batch, 1-ratio, batch.x.device)

def update_batch_after_augmentation(batch, mask):
    """
    在应用节点级增强后更新批次中的相关属性
    :param batch: 原始批次
    :param mask: 节点保留掩码
    :return: 更新后的批次
    """
    # 更新边索引
    row, col = batch.edge_index
    edge_mask = mask[row] & mask[col]
    
    # 创建新的节点ID映射
    old_to_new = torch.zeros(mask.size(0), dtype=torch.long, device=mask.device)
    old_to_new[mask] = torch.arange(mask.sum(), device=mask.device)
    
    # 更新边索引
    batch.edge_index = torch.stack([
        old_to_new[row[edge_mask]],
        old_to_new[col[edge_mask]]
    ], dim=0)
    
    # 更新边属性（如果存在）
    if hasattr(batch, 'edge_type') and batch.edge_type is not None:
        batch.edge_type = batch.edge_type[edge_mask]
    
    # 更新节点批次分配（如果存在）
    if hasattr(batch, 'batch') and batch.batch is not None:
        batch.batch = batch.batch[mask]
    
    return batch

'''
def gcl_train(args, model, device, train_loader, optimizer, model_dir=None, save_every=20):
    """
    训练GraphCL模型
    :param args: 参数配置
    :param model: GraphCL模型
    :param device: 设备
    :param train_loader: 预处理好的训练数据加载器
    :param optimizer: 优化器
    :param model_dir: 模型保存目录，如果为None则不保存
    :param save_every: 每多少个epoch保存一次模型
    :return: 训练得到的子图表示
    """
    # 从参数中提取数据集名称
    dataset_name = args.train_dataset.split('+')[0] if '+' in args.train_dataset else args.train_dataset
    
    # 获取模型参数字典，用于唯一标识模型
    model_params = get_model_params_dict(args, exclude_keys=['num_workers', 'gpu', 'epochs', 'batch_size', 'seed'])
    

       
    model.load_state_dict(torch.load('/home/hsl/baseline/GraphCL/models_graphcl/sandwich/graphcl_nh_4_tsk_regression_ngl_4_nhl_2_hd_144_us_0_bn_0_dp_0.3_best.pth'))
    print(f"成功加载模型: /home/hsl/baseline/GraphCL/models_graphcl/sandwich/graphcl_nh_4_tsk_regression_ngl_4_nhl_2_hd_144_us_0_bn_0_dp_0.3_best.pth")
    print("正在为已加载的模型计算子图嵌入...")
    
    # 计算子图嵌入
    model.eval()  # 设置为评估模式
    subgraph_embeddings = {}
    edge_id_mapping = {}
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(train_loader, desc="计算子图嵌入")):
            batch = batch.to(device)
            # 获取原始图的表示
            x = model.forward_cl(batch.x, batch.edge_index,
                                batch.edge_type if hasattr(batch, 'edge_type') else None,
                                batch.batch if hasattr(batch, 'batch') else None)
            
            # 保存子图表示
            if hasattr(batch, 'input_id'):
                for i, input_id in enumerate(batch.input_id):
                    edge_id = input_id.item()
                    subgraph_embeddings[edge_id] = x[i].detach().cpu()
                    if edge_id not in edge_id_mapping:
                        batch_size = batch.batch.max().item() + 1 if hasattr(batch, 'batch') and batch.batch is not None else len(batch.input_id)
                        edge_id_mapping[edge_id] = i + step * batch_size
    
    print(f"完成子图嵌入计算，共 {len(subgraph_embeddings)} 个子图")
    return subgraph_embeddings, edge_id_mapping