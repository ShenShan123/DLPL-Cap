import argparse
import torch
import numpy as np
from sram_dataset import performat_SramDataset, adaption_for_sgrl
from downstream_train import downstream_train
import os
import random
import gc
import sys
sys.path.append('..')
from utils_model_checkpoint import check_model_exists, save_model, load_model, get_model_params_dict
import datetime
from gcl_train import gcl_train
from sampling_and_pe import dataset_sampling_and_pe_calculation
from model import GraphHead
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, ResGatedGraphConv, GINConv, ChebConv, GINEConv, ClusterGCNConv
from gcl_model import graphcl

if __name__ == "__main__":
    # STEP 0: Parse Arguments ======================================================================= #
    parser = argparse.ArgumentParser(description="CircuitGPS_simple")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument("--train_dataset", type=str, default="sandwich+ultra8t", help="Names of training dataset.")
    parser.add_argument("--test_dataset", type=str, default="ssram+digtime+timing_ctrl+array_128_32_8t", help="Names of testing datasets.")
    parser.add_argument("--train_sample_rate", type=float, default=0.1, help="Sampling rate for training datasets.")
    parser.add_argument("--test_sample_rate", type=float, default=1.0, help="Sampling rate for testing datasets.")
    parser.add_argument("--add_tar_edge", type=int, default=0, help="0 or 1. Inject target edges into the graph.")
    # GraphCL arguments
    parser.add_argument('--cl_hid_dim', type=int, default=128, help='hidden_dim for contrastive learning')
    parser.add_argument('--cl_lr', type=float, default=0.001, help='learning rate for contrastive learning')
    parser.add_argument('--cl_epochs', type=int, default=200, help='Number of epochs for contrastive learning')
    parser.add_argument('--use_cl', type=int, default=1, help='Enable contrastive learning.')
    parser.add_argument('--aug1', type=str, default='dropN', help='First augmentation method')
    parser.add_argument('--aug2', type=str, default='permE', help='Second augmentation method')
    parser.add_argument('--aug_ratio1', type=float, default=0.2, help='Ratio for first augmentation')
    parser.add_argument('--aug_ratio2', type=float, default=0.2, help='Ratio for second augmentation')
    parser.add_argument('--gnn_type', type=str, default='ClusterGCN', help='GNN type for GraphCL')

    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay for contrastive learning')
    # CirGPS arguments
    parser.add_argument("--task", type=str, default="regression", help="Task type. 'classification' or 'regression'.")
    parser.add_argument("--loss", type=str, default='mse', help="The loss function. Could be 'mse', 'bmc', or 'gai'.")
    parser.add_argument("--noise_sigma", type=float, default=0.0001, help="The simga_noise of Balanced MSE (EQ 3.6).")
    parser.add_argument("--use_pe", type=int, default=0, help="Positional encoding. Defualt:False.")
    parser.add_argument("--num_hops", type=int, default=4, help="Number of hops in subgraph sampling.")
    parser.add_argument("--max_dist", type=int, default=350, help="The max values in DSPD.")
    parser.add_argument("--num_workers", type=int, default=4, help="The number of workers in data loaders.")
    parser.add_argument("--gpu", type=int, default=1, help="GPU index. Default: -1, using cpu.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="The batch size. Recommend <=64 to avoid memory issues.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--model", type=str, default='clustergcn', help="The gnn model. Could be 'clustergcn', 'resgatedgcn', 'gat', 'gcn', 'sage', 'gine'.")
    parser.add_argument("--num_gnn_layers", type=int, default=4, help="Number of GNN layers.")
    parser.add_argument("--num_head_layers", type=int, default=2, help="Number of head layers.")
    parser.add_argument("--hid_dim", type=int, default=144, help="Hidden layer dim.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout for neural networks.')
    parser.add_argument('--use_bn', type=int, default=0, help='0 or 1. Batch norm for neural networks.')
    parser.add_argument('--act_fn', default='relu', help='Activation function')
    parser.add_argument('--src_dst_agg', type=str, default='concat', help='The way to aggregate nodes. Can be `concat` or `add` or `pooladd` or `poolmean`.')
    parser.add_argument('--use_stats', type=int, default=0, help='0 or 1. Circuit statistics features.')
    args = parser.parse_args()

    # Syncronize all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu))
        print('Using GPU: {}'.format(args.gpu))
        # 清理GPU缓存
        torch.cuda.empty_cache()
        # 限制GPU内存增长
        torch.cuda.set_per_process_memory_fraction(0.5, device=args.gpu)  # 限制只使用50%的GPU内存
    else:
        device = torch.device("cpu")

    # 开启垃圾回收
    gc.enable()
    # 尝试设置较低的内存使用阈值
    gc.set_threshold(100, 5, 5)
    
    # 强制清理内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 创建日志目录和文件
    os.makedirs("./logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"./logs/{args.train_dataset}_to_{args.test_dataset}_{timestamp}.txt"
    
    # 重定向stdout到日志文件
    log_file = open(log_filename, 'w')
    original_stdout = sys.stdout
    
    # 使用Tee类来同时输出到控制台和文件
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    sys.stdout = Tee(original_stdout, log_file)
    
    print(f"日志文件已创建：{log_filename}")
    print(f"============= PID = {os.getpid()} ============= ")
    print("参数配置:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
        
    # STEP 1: Load Dataset =================================================================== #
    print("开始加载训练集...")
    train_dataset = performat_SramDataset(
        name=args.train_dataset, 
        dataset_dir='/local/hsl/datasets', 
        add_target_edges=args.add_tar_edge,
        neg_edge_ratio=0.5,
        to_undirected=True,
        sample_rates=args.train_sample_rate,
        task_type=args.task,
    )
    
    print("开始加载测试集...")
    test_dataset = performat_SramDataset(
        name=args.test_dataset, 
        dataset_dir='/local/hsl/datasets', 
        add_target_edges=args.add_tar_edge,
        neg_edge_ratio=0.5,
        to_undirected=True,
        sample_rates=args.test_sample_rate,
        task_type=args.task,
    )

    # STEP 2: 先进行子图采样，为每个链路生成子图 =========================================== #
    print("Running subgraph sampling...")
    (
        train_loader, val_loader, test_loaders,
        train_dspd_list, valid_dspd_list, test_dspd_dict,
    ) = dataset_sampling_and_pe_calculation(args, train_dataset, test_dataset)

    # STEP 3: 如果启用图对比学习，对采样的子图进行对比学习 ================================= #
    subgraph_embeddings = None
    edge_id_mapping = None
    
    if args.use_cl == 1:
        print("Running GraphCL on sampled subgraphs...")
        # 创建GraphCL模型保存目录
        model_dir = os.path.join("./models_graphcl", args.train_dataset.split('+')[0])
        os.makedirs(model_dir, exist_ok=True)
        
        # 创建GraphCL模型
        input_dim = train_dataset[0].x.shape[1]  # 获取节点特征维度
        hidden_dim = args.cl_hid_dim
        output_dim = args.cl_hid_dim
        
        # 选择合适的GNN模型
        if args.gnn_type == 'GCN':
            gnn = GCNConv(input_dim, hidden_dim)
        elif args.gnn_type == 'SAGE':
            gnn = SAGEConv(input_dim, hidden_dim)
        elif args.gnn_type == 'GAT':
            gnn = GATConv(input_dim, hidden_dim)
        elif args.gnn_type == 'ResGatedGCN':
            gnn = ResGatedGraphConv(input_dim, hidden_dim)
        elif args.gnn_type == 'GIN':
            nn_layer = torch.nn.Linear(input_dim, hidden_dim)
            gnn = GINConv(nn_layer)
        elif args.gnn_type == 'ClusterGCN':
            gnn = ClusterGCNConv(input_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported GNN type: {args.gnn_type}")
            
        model = graphcl(gnn).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.cl_lr, weight_decay=args.weight_decay)
        
        # 使用已经采样好的子图数据加载器进行图对比学习
        print(f"使用增强方法1: {args.aug1}, 比例: {args.aug_ratio1}")
        print(f"使用增强方法2: {args.aug2}, 比例: {args.aug_ratio2}")
        
        subgraph_embeddings, edge_id_mapping = gcl_train(
            args=args,
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            model_dir=model_dir,
            save_every=20
        )
        
    # STEP 4: 下游任务训练 ===================================================================== #
    dataset = {
        'train': train_dataset,
        'test': test_dataset
    }
    downstream_train(args, dataset, device, subgraph_embeddings, edge_id_mapping,
                    train_loader, val_loader, test_loaders,
                    train_dspd_list, valid_dspd_list, test_dspd_dict)
    
    # 恢复标准输出并关闭日志文件
    sys.stdout = original_stdout
    log_file.close()
    print(f"所有输出已保存到 {log_filename}")