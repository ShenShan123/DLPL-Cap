from sram_dataset import SealSramDataset
import torch
import numpy as np

class SramDataset_aug(SealSramDataset):
    def __init__(self,
                 name,
                 root,
                 add_target_edges=False,
                 neg_edge_ratio=1.0,
                 to_undirected=True,
                 sample_rates=[1.0],
                 task_type='classification',
                 transform=None,
                 pre_transform=None,
                 aug_methods=None,
                 aug_ratio=None):
        """
        # Extend the SealSramDataset class to support graph augmentation
        # :param aug_methods: List of augmentation methods to apply ('dropN', 'permE', 'maskN', 'subgraph')
        # :param aug_ratio: Augmentation ratio, indicating the proportion of nodes/edges to augment
        """
        self.aug_methods = aug_methods if aug_methods is not None else []
        self.aug_ratio = aug_ratio
        super(SramDataset_aug, self).__init__(
            name, root, add_target_edges, neg_edge_ratio,
            to_undirected, sample_rates, task_type, transform, pre_transform
        )

    def get(self, idx):
        # Get the original data
        data = super().get(idx)
        
        # Apply graph augmentations in sequence
        for aug_method in self.aug_methods:
            if aug_method == 'dropN':
                data = drop_nodes(data, self.aug_ratio)
            elif aug_method == 'permE':
                data = permute_edges(data, self.aug_ratio)
            elif aug_method == 'maskN':
                data = mask_nodes(data, self.aug_ratio)
            elif aug_method == 'subgraph':
                data = subgraph(data, self.aug_ratio)
            elif aug_method == 'addE':
                data = add_edges(data, self.aug_ratio)
            elif aug_method == 'semanticE':
                data = semantic_add_edges(data, self.aug_ratio)
            else:
                print(f'Unknown augmentation method: {aug_method}')
                assert False
            
        return data
    
def drop_nodes(data, aug_ratio):
    """
    Randomly delete a certain proportion of nodes and their connected edges
    :param data: SRAM graph data
    :param aug_ratio: Proportion of nodes to delete
    :return: Augmented graph data
    """
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    # Randomly select nodes to delete
    idx_perm = np.random.permutation(node_num)
    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    
    # Create a new index for the retained nodes
    idx_dict = {idx_nondrop[n]: n for n in range(idx_nondrop.shape[0])}

    # Keep edges that do not connect to the deleted nodes
    edge_index = data.edge_index.numpy()
    edge_mask = np.array([n for n in range(edge_num) if not (edge_index[0, n] in idx_drop or edge_index[1, n] in idx_drop)])

    # Reindex edges
    edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) 
                  if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    
    # Update data
    try:
        data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
        data.x = data.x[idx_nondrop]
        
        # If there is an edge attribute, it also needs to be updated
        if hasattr(data, 'edge_attr'):
            data.edge_attr = data.edge_attr[edge_mask]
        if hasattr(data, 'edge_type'):
            data.edge_type = data.edge_type[edge_mask]
            
        # Update node_attr (if it exists)
        if hasattr(data, 'node_attr'):
            data.node_attr = data.node_attr[idx_nondrop]
            
        # Update node_type (if it exists)
        if hasattr(data, 'node_type'):
            data.node_type = data.node_type[idx_nondrop]
    except:
        pass
        
    return data

def permute_edges(data, aug_ratio):
    """
    Randomly perturb a certain proportion of edges
    :param data: SRAM graph data
    :param aug_ratio: Proportion of edges to perturb
    :return: Augmented graph data
    """
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)
    
    # Randomly select edges to keep
    idx_delete = np.random.choice(edge_num, (edge_num - permute_num), replace=False)
    data.edge_index = data.edge_index[:, idx_delete]
    
    # Process edge attributes
    if hasattr(data, 'edge_attr'):
        data.edge_attr = data.edge_attr[idx_delete]
    if hasattr(data, 'edge_type'):
        data.edge_type = data.edge_type[idx_delete]
        
    return data

def mask_nodes(data, aug_ratio):
    """
    Randomly mask a certain proportion of node features
    :param data: SRAM graph data
    :param aug_ratio: Proportion of nodes to mask
    :return: Augmented graph data
    """
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    # Calculate the average value of node features as the mask value
    # For the SRAM dataset, consider the possibility of different node types
    if hasattr(data, 'node_attr') and data.node_attr is not None:
        # Use node_attr as features
        token = data.node_attr.mean(dim=0)
        idx_mask = np.random.choice(node_num, mask_num, replace=False)
        data.node_attr[idx_mask] = token
    else:
        # Use x as features
        token = data.x.mean(dim=0)
        idx_mask = np.random.choice(node_num, mask_num, replace=False)
        data.x[idx_mask] = token
        
    return data

def subgraph(data, aug_ratio):
    """
    Randomly extract a subgraph
    :param data: SRAM graph data
    :param aug_ratio: Proportion of nodes in the subgraph
    :return: Subgraph data
    """
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * aug_ratio)

    edge_index = data.edge_index.numpy()

    # Randomly select a starting node
    idx_sub = [np.random.randint(node_num, size=1)[0]]
    # Get the neighbors of the starting node
    idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

    # Breadth-first search to build the subgraph
    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh = idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

    # Get the nodes to delete and the nodes to keep
    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    
    # Reindex the retained nodes
    idx_dict = {idx_nondrop[n]: n for n in range(len(idx_nondrop))}
    
    # Keep the edges in the subgraph
    edge_mask = np.array([n for n in range(edge_num) if (edge_index[0, n] in idx_nondrop and edge_index[1, n] in idx_nondrop)])
    
    # Reindex the edges
    edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) 
                 if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    
    # Update data
    try:
        data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
        data.x = data.x[idx_nondrop]
        
        # If there is an edge attribute, it also needs to be updated
        if hasattr(data, 'edge_attr'):
            data.edge_attr = data.edge_attr[edge_mask]
        if hasattr(data, 'edge_type'):
            data.edge_type = data.edge_type[edge_mask]
            
        # Update node_attr (if it exists)
        if hasattr(data, 'node_attr'):
            data.node_attr = data.node_attr[idx_nondrop]
            
        # Update node_type (if it exists)
        if hasattr(data, 'node_type'):
            data.node_type = data.node_type[idx_nondrop]
    except:
        pass
        
    return data

def add_edges(data, aug_ratio):
    """
    Randomly add new edges between nodes that are not already connected
    :param data: SRAM graph data
    :param aug_ratio: Proportion of new edges to add relative to existing edges
    :return: Augmented graph data with additional edges
    """
    node_num, _ = data.x.size()
    edge_index = data.edge_index
    _, edge_num = edge_index.size()
    add_num = int(edge_num * aug_ratio)
    
    # Convert to set for fast lookup of existing edges
    existing_edges = set()
    for i in range(edge_num):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        existing_edges.add((src, dst))
    
    # Randomly add new edges
    new_edges = []
    attempts = 0
    max_attempts = add_num * 10  # Limit attempts to avoid infinite loop
    
    while len(new_edges) < add_num and attempts < max_attempts:
        # Randomly select two nodes
        src = np.random.randint(0, node_num)
        dst = np.random.randint(0, node_num)
        
        # Skip self-loops and existing edges
        if src != dst and (src, dst) not in existing_edges and (dst, src) not in existing_edges:
            new_edges.append([src, dst])
            existing_edges.add((src, dst))
        
        attempts += 1
    
    # If no new edges were found, return original data
    if not new_edges:
        return data
    
    # Add new edges to the graph
    new_edge_index = torch.tensor(new_edges, dtype=torch.long).t()
    data.edge_index = torch.cat([edge_index, new_edge_index], dim=1)
    
    # Handle edge attributes if they exist
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        # Create new edge attributes (using mean of existing attributes as default)
        if len(data.edge_attr) > 0:
            edge_attr_dim = data.edge_attr.size(1)
            default_attr = data.edge_attr.mean(dim=0).unsqueeze(0).repeat(len(new_edges), 1)
            data.edge_attr = torch.cat([data.edge_attr, default_attr], dim=0)
    
    # Handle edge types if they exist
    if hasattr(data, 'edge_type') and data.edge_type is not None:
        if len(data.edge_type) > 0:
            # Use most common edge type as default
            edge_type_counts = torch.bincount(data.edge_type)
            default_type = torch.argmax(edge_type_counts).item()
            default_types = torch.full((len(new_edges),), default_type, dtype=data.edge_type.dtype)
            data.edge_type = torch.cat([data.edge_type, default_types], dim=0)
    
    return data

def semantic_add_edges(data, aug_ratio):
    """
    根据电路设计规则添加新的耦合边：
    - 支持三种耦合类型：pin-to-pin, net-to-net, pin-to-net
    - 保持电路图的语义完整性
    
    :param data: SRAM图数据
    :param aug_ratio: 相对于现有边的新增边比例
    :return: 增强后的图数据，包含额外的语义合理的耦合边
    """
    # 如果没有节点类型信息，则返回原始数据
    if not hasattr(data, 'node_type'):
        print("Warning: semantic_add_edges requires node_type attribute, using original data")
        return data
        
    node_num, _ = data.x.size()
    edge_index = data.edge_index
    _, edge_num = data.edge_index.size()
    add_num = int(edge_num * aug_ratio)
    
    # 获取节点类型
    node_types = data.node_type
    
    # 根据sram_dataset.py中的信息识别节点类型
    # g._n2type = {'device': 0, 'pin': 1, 'net': 2}
    DEVICE_TYPE = 0  # 器件节点
    PIN_TYPE = 1     # 引脚节点
    NET_TYPE = 2     # 网络节点
    
    # 边类型定义
    # 根据sram_dataset.py中的g._e2type定义：
    # ('device', 'device-pin', 'pin'): 0  # 电路拓扑连接
    # ('pin', 'pin-net', 'net'): 1        # 电路拓扑连接
    # ('pin', 'cc_p2n', 'net'): 2         # pin到net的耦合电容边
    # ('pin', 'cc_p2p', 'pin'): 3         # pin到pin的耦合电容边
    # ('net', 'cc_n2n', 'net'): 4         # net到net的耦合电容边
    
    P2N_COUPLING_TYPE = 2  # pin到net的耦合边类型
    P2P_COUPLING_TYPE = 3  # pin到pin的耦合边类型
    N2N_COUPLING_TYPE = 4  # net到net的耦合边类型
    
    # 找到所有pin节点和net节点
    pin_indices = torch.nonzero(node_types == PIN_TYPE).squeeze().tolist()
    net_indices = torch.nonzero(node_types == NET_TYPE).squeeze().tolist()
    
    # 确保索引是列表
    if not isinstance(pin_indices, list):
        pin_indices = [pin_indices]
    if not isinstance(net_indices, list):
        net_indices = [net_indices]
        
    # 检查是否有足够的节点
    if len(pin_indices) < 1 or len(net_indices) < 1:
        print("Warning: Not enough pin/net nodes found for semantic edge augmentation")
        return data
        
    # 找到所有现有的边，用于快速查找
    existing_edges = set()
    for i in range(edge_num):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        existing_edges.add((src, dst))
        existing_edges.add((dst, src))  # 考虑无向图
    
    # 添加新的耦合边
    new_edges = []
    new_edge_types = []
    attempts = 0
    max_attempts = add_num * 10
    
    while len(new_edges) < add_num and attempts < max_attempts:
        # 随机选择耦合边类型
        coupling_type = np.random.choice([P2N_COUPLING_TYPE, P2P_COUPLING_TYPE, N2N_COUPLING_TYPE])
        
        if coupling_type == P2N_COUPLING_TYPE and len(pin_indices) > 0 and len(net_indices) > 0:
            # pin到net的耦合边
            src = np.random.choice(pin_indices)
            dst = np.random.choice(net_indices)
            edge_type = P2N_COUPLING_TYPE
        elif coupling_type == P2P_COUPLING_TYPE and len(pin_indices) >= 2:
            # pin到pin的耦合边
            idx1, idx2 = np.random.choice(len(pin_indices), 2, replace=False)
            src = pin_indices[idx1]
            dst = pin_indices[idx2]
            edge_type = P2P_COUPLING_TYPE
        elif coupling_type == N2N_COUPLING_TYPE and len(net_indices) >= 2:
            # net到net的耦合边
            idx1, idx2 = np.random.choice(len(net_indices), 2, replace=False)
            src = net_indices[idx1]
            dst = net_indices[idx2]
            edge_type = N2N_COUPLING_TYPE
        else:
            attempts += 1
            continue
            
        # 跳过自环和已存在的边
        if src != dst and (src, dst) not in existing_edges:
            new_edges.append([src, dst])
            new_edge_types.append(edge_type)
            existing_edges.add((src, dst))
            existing_edges.add((dst, src))
            
        attempts += 1
    
    # 如果没有添加新边，返回原始数据
    if not new_edges:
        return data
    
    # 将新边添加到图中
    new_edge_index = torch.tensor(new_edges, dtype=torch.long).t()
    data.edge_index = torch.cat([edge_index, new_edge_index], dim=1)
    
    # 处理边属性（如果存在）
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        if len(data.edge_attr) > 0:
            # 为不同类型的耦合边设置不同的属性
            all_new_attrs = []
            
            for i, edge_type in enumerate(new_edge_types):
                # 找到相同类型的现有边作为模板
                template_mask = []
                for j in range(edge_num):
                    if hasattr(data, 'edge_type') and data.edge_type[j].item() == edge_type:
                        template_mask.append(j)
                
                if template_mask:
                    # 使用相同类型边的属性平均值
                    template_attrs = data.edge_attr[template_mask]
                    attr = template_attrs.mean(dim=0).unsqueeze(0)
                else:
                    # 回退到使用所有边的平均属性
                    attr = data.edge_attr.mean(dim=0).unsqueeze(0)
                    
                all_new_attrs.append(attr)
                
            if all_new_attrs:
                default_attr = torch.cat(all_new_attrs, dim=0)
                data.edge_attr = torch.cat([data.edge_attr, default_attr], dim=0)
    
    # 处理边类型（如果存在）
    if hasattr(data, 'edge_type') and data.edge_type is not None:
        if len(data.edge_type) > 0:
            # 为每条边设置对应的耦合类型
            new_edge_types_tensor = torch.tensor(new_edge_types, dtype=data.edge_type.dtype)
            data.edge_type = torch.cat([data.edge_type, new_edge_types_tensor], dim=0)
    
    return data