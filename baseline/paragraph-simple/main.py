import argparse
import torch
import numpy as np
from sram_dataset import performat_SramDataset
from downstream_train import downstream_link_pred
import os
import random
import gc
import sys
import datetime

if __name__ == "__main__":
    # STEP 0: Parse Arguments ======================================================================= #
    parser = argparse.ArgumentParser(description="CircuitGPS_simple")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument("--train_dataset", type=str, default="sandwich+ultra8t", help="Name of training dataset.")
    parser.add_argument("--test_dataset", type=str, default="ssram+digtime+timing_ctrl+array_128_32_8t", help="Name of test dataset.")
    parser.add_argument("--task", type=str, default="regression", help="Task type. 'classification' or 'regression'.")
    parser.add_argument("--max_dist", type=int, default=350, help="The max values in DSPD.")
    parser.add_argument("--num_workers", type=int, default=4, help="The number of workers in data loaders.")
    parser.add_argument("--gpu", type=int, default=3, help="GPU index. Default: -1, using cpu.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size.")
    parser.add_argument("--lr", type=float, default=0.00005, help="Learning rate.")
    parser.add_argument("--num_gnn_layers", type=int, default=4, help="Number of GNN layers.")  # The number of FC layers were set to 4 for the net parasitic capacitance model
    parser.add_argument("--num_head_layers", type=int, default=2, help="Number of head layers.")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dim.")  #the dimension of embedding is set to F = 32
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout for neural networks.')
    parser.add_argument('--use_bn', type=int, default=0, help='Batch norm for neural networks.')
    parser.add_argument('--act_fn', default='relu', help='Activation function')
    parser.add_argument('--src_dst_agg', default='concat', help='The way to aggregate nodes. Can be "concat" or "add" or "pool".')
    parser.add_argument('--num_hops',type=int,default=4,help='Number of hops.')
    parser.add_argument('--to_undirected', type=int, default=1, help='Whether to convert the graph to undirected graph.')
    parser.add_argument('--train_sample_rate', type=float, default=0.1, help='Sampling rate for training datasets.')
    parser.add_argument('--test_sample_rate', type=float, default=1.0, help='Sampling rate for testing datasets.')
    parser.add_argument('--use_ensemble', type=int, default=0, help='Whether to use ensemble model for predictions.')
    parser.add_argument('--num_ensemble', type=int, default=3, help='Number of models in the ensemble.')
    parser.add_argument('--ensemble_thresholds', type=str, default='0.33,0.66', 
                        help='Comma-separated max prediction thresholds for Algorithm 2 ensemble strategy.')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.use_ensemble:
        args.ensemble_thresholds = [float(x) for x in args.ensemble_thresholds.split(',')]
        print(f"Using ensemble model with {args.num_ensemble} models and thresholds: {args.ensemble_thresholds}")

    if args.gpu != -1 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu))
        # 清理GPU缓存
        torch.cuda.empty_cache()
        # 限制GPU显存使用为12GB（一半）
        torch.cuda.set_per_process_memory_fraction(0.5, device=args.gpu)
        print(f'Using GPU: {args.gpu}, 已限制GPU显存使用为50%')
    else:
        device = torch.device("cpu")
        
    # 启用垃圾回收和设置较低的阈值
    gc.enable()
    gc.set_threshold(100, 5, 5)
    gc.collect()

    # 优化batch size，避免内存问题
    if args.batch_size > 128:
        print(f"Warning: 降低batch_size从{args.batch_size}到128以避免内存问题")
        args.batch_size = 128
    
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

    train_dataset = performat_SramDataset(
        name=args.train_dataset, 
        dataset_dir='/local/hsl/datasets-para', 
        neg_edge_ratio=0.3,
        to_undirected=args.to_undirected,
        sample_rates=args.train_sample_rate,
        task_type=args.task,
    )
    
    test_dataset = performat_SramDataset(
        name=args.test_dataset, 
        dataset_dir='/local/hsl/datasets-para', 
        neg_edge_ratio=0.3,
        to_undirected=args.to_undirected,
        sample_rates=args.test_sample_rate,
        task_type=args.task,
    )
    
    dataset = {
        'train': train_dataset,
        'test': test_dataset
    }

    downstream_link_pred(args, dataset, device)
    
    # 恢复标准输出并关闭日志文件
    sys.stdout = original_stdout
    log_file.close()
    print(f"所有输出已保存到 {log_filename}")