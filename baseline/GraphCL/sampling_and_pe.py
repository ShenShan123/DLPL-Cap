import torch
import networkx as nx
from torch_geometric.utils import to_dense_adj, to_networkx
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import pickle
import gc  # 导入垃圾回收模块

def get_double_spd(data, anchor_indices, max_dist):
    """
    Compute shortest path distances from multiple anchor nodes to all other nodes
    in an undirected graph using NetworkX.
    
    Args:
        data (torch_geometric.data.Data): Input graph data
        anchor_indices (torch.Tensor or list): Indices of anchor nodes (shape: [M])
    
    Returns:
        torch.Tensor: Tensor of shortest path distances (shape: [num_nodes, M])
    """
    # Convert PyG data to NetworkX undirected graph once
    G = to_networkx(data, to_undirected=True)
    num_nodes = data.num_nodes
    M = len(anchor_indices)
    
    # Initialize distance matrix with -1 (unreachable)
    distances = torch.full((num_nodes, M), max_dist, dtype=torch.long)

    # Convert to list if given as tensor
    if isinstance(anchor_indices, torch.Tensor):
        anchor_indices = anchor_indices.tolist()

    for i, anchor in enumerate(anchor_indices):
        if anchor not in G:
            raise ValueError(f"Anchor node {anchor} not found in graph")
            
        # Get shortest paths using BFS
        shortest_lengths = nx.single_source_shortest_path_length(G, anchor)
        
        # Fill distances for this anchor column
        for node, dist in shortest_lengths.items():
            distances[node, i] = dist if dist < max_dist else max_dist
    # print(distances)
    return distances

def pe_encoding_for_graph(
        args, graph, edge_label_index, edge_label, processed_pe_path=None,
    ):
    """
    With a given graph in dataset, do subgraph sampling and 
    then calculate the DSPD for the sampled subgraph.
    Args:
        args (argparse.Namespace): The arguments
        graph (torch_geometric.data.Data): The graph
        graph_name (str): The name of the graph
        edge_label_index (torch.Tensor): The edge label index
        edge_label (torch.Tensor): The edge label
        processed_pe_path (str): The path to save the DSPD per batch
    Return:
        loader: The loader with 'batch_size' for mini-batch training
        batch_dspd_list: The DSPDs of batches coming from the loader.
    """
    num_neighbors = -1
    path_exist = os.path.exists(processed_pe_path)
    
    # 检查是否存在临时缓存文件夹
    temp_dir = os.path.dirname(processed_pe_path) + "/temp_dspd"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 首先清理所有现有的临时文件
    for f in os.listdir(temp_dir):
        if f.endswith('.pkl'):
            os.remove(os.path.join(temp_dir, f))

    ## If we do not use PE, just return the loader and an empty list
    if (not args.use_pe) or path_exist:
        ## The actual loader used in mini-batch training
        loader = LinkNeighborLoader(
            graph,
            num_neighbors=args.num_hops * [num_neighbors],
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            subgraph_type='bidirectional',
            disjoint=True,
            batch_size=args.batch_size,
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=False,
        )
        if path_exist and args.use_pe:
            print("Found existing file of dspd_per_batch!")
            print(f"Loading from {processed_pe_path}")

            with open (processed_pe_path, 'rb') as fp:
                dspd_per_batch = pickle.load(fp)
        
        else:
            dspd_per_batch = [None] * edge_label.size(0)
        return loader, dspd_per_batch
    
    # 分批处理以节省内存，一次最多处理1000个子图
    total_edges = edge_label_index.size(1)
    batch_size = min(100, total_edges)  # 将批量大小减少到100
    num_batches = (total_edges + batch_size - 1) // batch_size
    dspd_per_subg = [None] * total_edges
    
    for batch_idx in range(num_batches):
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_edges)
        print(f"Processing batch {batch_idx+1}/{num_batches} (edges {start_idx} to {end_idx})")
        
        # 为当前批次创建子加载器
        batch_edge_label_index = edge_label_index[:, start_idx:end_idx]
        batch_edge_label = edge_label[start_idx:end_idx]
        
        ## Create a LinkNeighborLoader for subgraph sampling.
        ## For each edge_label_index, we sample a 'num_hops' subgraph.
        ## NOTE: This loader is only used for PE calculation.
        loader = LinkNeighborLoader(
            graph,
            num_neighbors=args.num_hops * [num_neighbors],
            edge_label_index=batch_edge_label_index,
            edge_label=batch_edge_label,
            subgraph_type='bidirectional',
            disjoint=True,
            batch_size=1, ## batch_size is always 1
            shuffle=False, 
            num_workers=1,  # 减少worker数
            pin_memory=False,
        )

        ## Calculate the SPD for each batch
        for i, subgraph in enumerate(tqdm(
            loader, 
            desc=f"{graph.name}: Subgraph sampling and DSPD calculation (batch {batch_idx+1}/{num_batches})"
        )):
            global_idx = start_idx + i
            dspd = get_double_spd(
                subgraph,
                ## src and dst nodes in edge_label_index are always 
                ## the first 2 nodes in the subgraph.
                anchor_indices=[0, 1], max_dist=args.max_dist,
            )
            assert dspd.size(0) == subgraph.num_nodes
            dspd_per_subg[global_idx] = dspd
            
            # 每处理10个子图就清理一次内存
            if i % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            # 每处理单个子图就清理一次内存，更激进地释放内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 删除不再需要的变量
            del subgraph, dspd
            
        # 清理loader
        del loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    ## The actual loader used in mini-batch training
    loader = LinkNeighborLoader(
        graph,
        num_neighbors=args.num_hops * [num_neighbors],
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        subgraph_type='bidirectional',
        disjoint=True,
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=False,
    )

    dspd_per_batch = [None] * edge_label.size(0)
    
    ## match the DSPDs of subgraphs back to the data batches
    for b, batch in enumerate(
        tqdm(loader, desc='Matching back to batches', leave=False)
    ):
        batched_dspd = torch.empty(
            (batch.num_nodes, 2), dtype=torch.long).fill_(args.max_dist)
        ## For each batrch, we have:
        ## batch.edge_label.size(0) == batch.edge_label_index.size(1)
        ## batch.batch.max()+1 == batch.input_id.size(0) == \
        num_subgraphs = batch.input_id.size(0)

        for i in range(num_subgraphs):
            subg_node_mask = batch.batch == i
            ## global subgraph id is the id of the sampled 'edge_label_index'
            global_subg_id = batch.input_id[i]
            batched_dspd[subg_node_mask] = dspd_per_subg[global_subg_id]
        
        ## store the dspd for each batch
        dspd_per_batch[b] = batched_dspd
        
        # 每处理10个批次清理一次内存
        if b % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    ## save dspd_per_batch to file
    print(f"Saving dspd_per_batch to {processed_pe_path}")
    with open(processed_pe_path, 'wb') as fp:
        pickle.dump(dspd_per_batch, fp)

    return loader, dspd_per_batch

def dataset_sampling_and_pe_calculation(args, train_dataset, test_dataset):
    """ 
    Sampling subgraphs for each graph in dataset and 
    calculate the PE for each sampled subgraph.
    Args:
        args (argparse.Namespace): The arguments
        train_dataset (torch_geometric.data.InMemoryDataset): The training dataset
        test_dataset (torch_geometric.data.InMemoryDataset): The testing dataset
    Return:
        train_loader, val_loader, test_loaders, 
        train_subgraph_dspd_list, valid_subgraph_dspd_list, test_subgraph
    """
    # 合并训练数据集中的图
    train_graphs = []
    total_edge_labels = 0
    total_nodes = 0
    total_edges = 0
    
    # 收集所有的训练图
    for i in range(len(train_dataset)):
        train_graph = train_dataset[i]
        train_graphs.append(train_graph)
        total_edge_labels += train_graph.edge_label.size(0)
        total_nodes += train_graph.num_nodes
        total_edges += train_graph.edge_index.size(1)

    # 创建合并后的图
    merged_graph = train_graphs[0].__class__()
    
    # 合并节点特征
    merged_graph.x = torch.cat([g.x for g in train_graphs], dim=0)
    
    # 合并边索引，需要调整索引
    edge_index_list = []
    node_offset = 0
    for g in train_graphs:
        edge_index = g.edge_index.clone()
        edge_index += node_offset
        edge_index_list.append(edge_index)
        node_offset += g.num_nodes
    merged_graph.edge_index = torch.cat(edge_index_list, dim=1)
    
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
    
    # 设置其他属性
    merged_graph.num_nodes = total_nodes
    merged_graph.name = "merged_train_graph"
    
    ## get split for validation
    train_ind, val_ind = train_test_split(
        np.arange(merged_graph.edge_label.size(0)), 
        test_size=0.2, shuffle=True, #stratify=stratify,
    )
    train_ind = torch.tensor(train_ind, dtype=torch.long)
    val_ind = torch.tensor(val_ind, dtype=torch.long)

    train_edge_label_index = merged_graph.edge_label_index[:, train_ind]
    train_edge_label = merged_graph.edge_label[train_ind]
    dspd_name = f'_h{args.num_hops}_seed{args.seed}_train.dspd'
    processed_pe_path = os.path.join(
        os.path.dirname(train_dataset.processed_paths[0]), 
        dspd_name
    )
    train_loader, train_dspd_list = pe_encoding_for_graph(
        args, merged_graph, train_edge_label_index, train_edge_label, processed_pe_path
    )

    val_edge_label_index = merged_graph.edge_label_index[:, val_ind]
    val_edge_label = merged_graph.edge_label[val_ind]
    dspd_name = f'_h{args.num_hops}_seed{args.seed}_val.dspd'
    processed_pe_path = os.path.join(
        os.path.dirname(train_dataset.processed_paths[0]), 
        dspd_name
    )
    val_loader, valid_dspd_list = pe_encoding_for_graph(
        args, merged_graph, val_edge_label_index, val_edge_label, processed_pe_path
    )

    ## test data come from the rest datasets
    test_loaders = []
    test_dspd_dict = {}
    for graph_idx in range(len(test_dataset)):
        test_graph = test_dataset[graph_idx]
        test_edge_label_index = test_graph.edge_label_index
        test_edge_label = test_graph.edge_label
        dspd_name = f'_h{args.num_hops}_seed{args.seed}_test_{graph_idx}.dspd'
        processed_pe_path = os.path.join(
            os.path.dirname(test_dataset.processed_paths[0]), 
            dspd_name
        )
        test_loader, test_dspd_list = pe_encoding_for_graph(
            args, test_graph, test_edge_label_index, test_edge_label, processed_pe_path
        )
        test_loaders.append(test_loader)
        test_dspd_dict[graph_idx] = test_dspd_list

    return (
        train_loader, val_loader, test_loaders,
        train_dspd_list, valid_dspd_list, test_dspd_dict,
    )