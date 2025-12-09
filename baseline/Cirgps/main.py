import argparse
import torch
import numpy as np
from sram_dataset import performat_SramDataset, adaption_for_sgrl
from downstream_train import downstream_train
import os
import random
import sys
import gc
import datetime
import psutil
import tempfile

# 设置自定义临时目录
os.environ['TMPDIR'] = '/data1/tmp'
if not os.path.exists('/data1/tmp'):
    os.makedirs('/data1/tmp', exist_ok=True)
tempfile.tempdir = '/data1/tmp'

if __name__ == "__main__":
    # STEP 0: Parse Arguments ======================================================================= #
    parser = argparse.ArgumentParser(description="CircuitGPS_simple")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument("--train_dataset", type=str, default="sandwich+ultra8t", help="Name of training dataset.")
    parser.add_argument("--test_dataset", type=str, default="ssram+digtime+timing_ctrl+array_128_32_8t", help="Name of test dataset.")
    parser.add_argument("--add_tar_edge", type=int, default=0, help="0 or 1. Inject target edges into the graph.")
    parser.add_argument("--train_sample_rate", type=float, default=0.1, help="Sampling rate for training datasets.")
    parser.add_argument("--test_sample_rate", type=float, default=1.0, help="Sampling rate for testing datasets.")
    parser.add_argument('--u', type=float, default=0.5, help='Parameter u for experimentation.')
    # parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
    # parser.add_argument('--num_hop', type=int, default=1, help='num_hop')
    # parser.add_argument('--trials', type=int, default=20, help='trials')
    # CirGPS arguments
    parser.add_argument("--task", type=str, default="regression", help="Task type. 'classification' or 'regression'.")
    parser.add_argument("--use_pe", type=int, default=1, help="Positional encoding. Defualt: True.")
    parser.add_argument("--num_hops", type=int, default=1, help="Number of hops in subgraph sampling.")
    parser.add_argument("--max_dist", type=int, default=350, help="The max values in DSPD.")
    parser.add_argument("--num_workers", type=int, default=0, help="The number of workers in data loaders.")
    parser.add_argument("--gpu", type=int, default=1, help="GPU index. Default: -1, using cpu.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--model", type=str, default='clustergcn', help="The gnn model. Could be 'clustergcn', 'resgatedgcn', 'gat', 'gcn', 'sage', 'gine'.")
    parser.add_argument("--num_gnn_layers", type=int, default=4, help="Number of GNN layers.")
    parser.add_argument("--num_head_layers", type=int, default=2, help="Number of head layers.")
    parser.add_argument("--hid_dim", type=int, default=144, help="Hidden layer dim.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout for neural networks.')
    parser.add_argument('--use_bn', type=int, default=1, help='0 or 1. Batch norm for neural networks.')
    parser.add_argument('--act_fn', default='relu', help='Activation function')
    parser.add_argument('--src_dst_agg', type=str, default='concat', help='The way to aggregate nodes. Can be `concat` or `add` or `pooladd` or `poolmean`.')
    parser.add_argument('--use_stats', type=int, default=0, help='0 or 1. Circuit statistics features.')
    # 内存管理参数
    parser.add_argument('--mem_fraction', type=float, default=0.5, help='GPU显存使用比例限制 (0-1)')
    parser.add_argument('--pe_chunk_size', type=int, default=2000, help='PE计算时的分块大小')
    parser.add_argument('--skip_pe_calculation', type=int, default=0, help='是否跳过PE计算 (0或1)')
    parser.add_argument('--low_memory_mode', type=int, default=0, help='启用低内存模式 (0或1)')
    args = parser.parse_args()

    # 创建日志目录
    os.makedirs("./logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"./logs/{args.train_dataset}_to_{args.test_dataset}_{timestamp}.txt"
    
    # 将所有print输出重定向到日志文件
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

    # 内存使用监控函数
    def print_memory_usage():
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / (1024 ** 3)
        print(f"当前内存使用: {memory_gb:.2f} GB")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_mem_alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
                gpu_mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                print(f"GPU {i} 显存使用: {gpu_mem_alloc:.2f} GB (已分配), {gpu_mem_reserved:.2f} GB (已预留)")

    # Syncronize all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 内存优化：启用垃圾回收
    gc.enable()
    gc.set_threshold(100, 5, 5)
    print("已启用主动垃圾回收机制")
    
    # 检查并打印当前内存使用情况
    print("初始内存状态:")
    print_memory_usage()

    # Check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu))
        print('Using GPU: {}'.format(args.gpu))
        # 清理GPU缓存
        torch.cuda.empty_cache()
        # 限制GPU显存使用
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, device=args.gpu)
        print(f'已限制GPU显存使用为{args.mem_fraction*100}%')
    else:
        device = torch.device("cpu")

    print(f"============= PID = {os.getpid()} ============= ")
    print(f"Parameter u: {args.u}")
    print("参数配置:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    # 强制进行一次垃圾回收
    gc.collect()
    
    # 低内存模式下调整参数
    if args.low_memory_mode:
        print("启用低内存模式")
        # 减少batch_size
        original_batch_size = args.batch_size
        args.batch_size = min(args.batch_size, 128)
        print(f"调整batch_size: {original_batch_size} -> {args.batch_size}")
        
        # 减少num_workers
        original_num_workers = args.num_workers
        args.num_workers = min(args.num_workers, 8)
        print(f"调整num_workers: {original_num_workers} -> {args.num_workers}")
        
        # 如果低内存模式启用且没有显式指定PE计算块大小，设置较小的默认值
        if args.pe_chunk_size == 2000:
            args.pe_chunk_size = 1000
            print(f"调整PE计算块大小为: {args.pe_chunk_size}")
            
        # 传递内存管理相关参数给PE计算函数
        os.environ["PE_CHUNK_SIZE"] = str(args.pe_chunk_size)
        os.environ["SKIP_PE_CALCULATION"] = str(args.skip_pe_calculation)
        os.environ["LOW_MEMORY_MODE"] = str(args.low_memory_mode)
    
    # STEP 1: Load Dataset =================================================================== #
    print("开始加载数据集...")
    train_dataset = performat_SramDataset(
        name=args.train_dataset, 
        dataset_dir='/local/hsl/datasets-cirgps', 
        add_target_edges=args.add_tar_edge,
        neg_edge_ratio=0.5,
        to_undirected=True,
        sample_rates=args.train_sample_rate,
        task_type=args.task,
    )
    
    print("训练集加载完成，检查内存使用:")
    print_memory_usage()
    gc.collect()
    
    test_dataset = performat_SramDataset(
        name=args.test_dataset, 
        dataset_dir='/local/hsl/datasets-cirgps', 
        add_target_edges=args.add_tar_edge,
        neg_edge_ratio=0.5,
        to_undirected=True,
        sample_rates=args.test_sample_rate,
        task_type=args.task,
    )
    
    print("测试集加载完成，检查内存使用:")
    print_memory_usage()
    gc.collect()
    
    # STEP 2-3: If you do graph contrastive learning, you should add the code here =========== #
    
    # STEP 4: Training Epochs ================================================================ #
    # No graph contrastive learning, no initail embeddings
    # embeds = torch.zeros(( train_graph.num_nodes, hid_dim))
    dataset = {
        'train': train_dataset,
        'test': test_dataset
    }
    
    print("开始训练过程...")
    downstream_train(args, dataset, device) 
    
    # 关闭输出文件并恢复标准输出
    sys.stdout = original_stdout
    log_file.close()
    print(f"所有输出已保存到日志文件 {log_filename}") 