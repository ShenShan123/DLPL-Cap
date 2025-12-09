import torch
import numpy as np
from torch_geometric.loader import LinkNeighborLoader
from sklearn.model_selection import train_test_split

def dataset_sampling(args, train_dataset, test_dataset):
    """
    Args:
        args (argparse.Namespace): The arguments
        dataset (list): List containing datasets. First one is the training dataset, followed by test datasets
    Return:
        train_loader, val_loader, test_loaders
    """
   
    
    # 合并训练数据集中的图
    train_graphs = []
    total_edge_labels = 0
    total_nodes = 0
    total_edges = 0
    
    # 收集所有的训练图
    for i in range(len(train_dataset.names)):
        train_graph = train_dataset[i]
        train_graphs.append(train_graph)
        total_edge_labels += train_graph.edge_label.size(0)
        total_nodes += train_graph.num_nodes
        total_edges += train_graph.edge_index.size(1)

    # 创建合并后的图
    merged_graph = train_graphs[0].__class__()
    
    # 合并节点特征
    merged_graph.x = torch.cat([g.x for g in train_graphs], dim=0)
    
    # 合并节点属性
    if hasattr(train_graphs[0], 'node_attr'):
        merged_graph.node_attr = torch.cat([g.node_attr for g in train_graphs], dim=0)
    
    # 合并边索引，需要调整索引
    edge_index_list = []
    edge_type_list = []
    node_offset = 0
    for g in train_graphs:
        edge_index = g.edge_index.clone()
        edge_index += node_offset
        edge_index_list.append(edge_index)
        edge_type_list.append(g.edge_type)
        node_offset += g.num_nodes
    merged_graph.edge_index = torch.cat(edge_index_list, dim=1)
    merged_graph.edge_type = torch.cat(edge_type_list, dim=0)
    
    # 合并边标签索引
    edge_label_index_list = []
    node_offset = 0
    for g in train_graphs:
        edge_label_index = g.edge_label_index.clone()
        edge_label_index += node_offset
        edge_label_index_list.append(edge_label_index)
        node_offset += g.num_nodes
    merged_graph.edge_label_index = torch.cat(edge_label_index_list, dim=1)
    
    # 合并边标签
    merged_graph.edge_label = torch.cat([g.edge_label for g in train_graphs], dim=0)
    
    # 合并节点类型映射
    if hasattr(train_graphs[0], '_n2type'):
        merged_graph._n2type = train_graphs[0]._n2type.copy()
        merged_graph._num_ntypes = train_graphs[0]._num_ntypes
    
    # 合并边类型映射
    if hasattr(train_graphs[0], '_e2type'):
        merged_graph._e2type = train_graphs[0]._e2type.copy()
        merged_graph._num_etypes = train_graphs[0]._num_etypes
    
    # 设置其他属性
    merged_graph.num_nodes = total_nodes
    merged_graph.name = "merged_train_graph"
    
    # 获取训练和验证集拆分
    train_ind, val_ind = train_test_split(
        np.arange(merged_graph.edge_label.size(0)), 
        test_size=0.2, shuffle=True, 
    )
    train_ind = torch.tensor(train_ind, dtype=torch.long)
    val_ind = torch.tensor(val_ind, dtype=torch.long)

    train_edge_label_index = merged_graph.edge_label_index[:, train_ind]
    train_edge_label = merged_graph.edge_label[train_ind]

    # 创建训练数据加载器
    train_loader = LinkNeighborLoader(
        merged_graph,
        num_neighbors=args.num_hops * [-1],  # -1 represents all neighbors.
        edge_label_index=train_edge_label_index,
        edge_label=train_edge_label,
        subgraph_type='bidirectional',
        disjoint=True,
        batch_size=args.batch_size,
        shuffle=True,  # shuffle in training.
        num_workers=args.num_workers,
    )

    # 创建验证数据加载器
    val_edge_label_index = merged_graph.edge_label_index[:, val_ind]
    val_edge_label = merged_graph.edge_label[val_ind]
   
    val_loader = LinkNeighborLoader(
        merged_graph,
        num_neighbors=args.num_hops * [-1],
        edge_label_index=val_edge_label_index,
        edge_label=val_edge_label,
        subgraph_type='bidirectional',
        disjoint=True,
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
    )

    # 创建测试数据加载器
    test_loaders = {}
    
    
    
    # 为每个测试数据集创建数据加载器
    for i in range(len(test_dataset.names)):
        test_graph = test_dataset[i]
        graph_name = test_graph.name
       
        test_loaders[graph_name] = LinkNeighborLoader(
            test_graph,
            num_neighbors=args.num_hops * [-1],
            edge_label_index=test_graph.edge_label_index,
            edge_label=test_graph.edge_label,
            subgraph_type='bidirectional',
            disjoint=True,
            batch_size=args.batch_size,
            shuffle=False,  
            num_workers=args.num_workers,
        )

    return train_loader, val_loader, test_loaders
