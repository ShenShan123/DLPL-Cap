import argparse
import torch
import numpy as np
from sram_dataset import performat_SramDataset
from downstream_train import downstream_link_pred
import os
import random
import datetime
import sys

if __name__ == "__main__":
    # STEP 0: Parse Arguments ======================================================================= #
    parser = argparse.ArgumentParser(description="CircuitGPS_simple")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument("--train_dataset", type=str, default="sandwich+ultra8t", help="Name of training dataset.")
    parser.add_argument("--test_dataset", type=str, default="ssram+digtime+timing_ctrl+array_128_32_8t", help="Name of test dataset.")
    parser.add_argument("--task", type=str, default="regression", help="Task type. 'classification' or 'regression'.")
    parser.add_argument("--max_dist", type=int, default=350, help="The max values in DSPD.")
    parser.add_argument("--num_workers", type=int, default=4, help="The number of workers in data loaders.")
    parser.add_argument("--gpu", type=int, default=1, help="GPU index. Default: -1, using cpu.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size.")
    parser.add_argument("--lr", type=float, default=0.00005, help="Learning rate.")
    parser.add_argument("--num_gnn_layers", type=int, default=4, help="Number of GNN layers.")  # The number of FC layers were set to 4 for the net parasitic capacitance model
    parser.add_argument("--num_head_layers", type=int, default=2, help="Number of head layers.")
    parser.add_argument("--hid_dim", type=int, default=32, help="Hidden layer dim.")  #the dimension of embedding is set to F = 32
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout for neural networks.')
    parser.add_argument('--use_bn', type=int, default=0, help='Batch norm for neural networks.')
    parser.add_argument('--act_fn', default='relu', help='Activation function')
    parser.add_argument('--src_dst_agg', default='concat', help='The way to aggregate nodes. Can be "concat" or "add" or "pool".')
    parser.add_argument('--num_hops',type=int,default=4,help='Number of hops.')
    parser.add_argument('--to_undirected', type=int, default=1, help='Whether to convert the graph to undirected graph.')
    parser.add_argument('--use_stats', type=int, default=1, help='Whether to use circuit statistics encoder.')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs.')
    parser.add_argument('--use_focal_loss', type=int, default=0, help='Whether to use Focal Loss for classification tasks, which is more effective for unbalanced datasets.')
    parser.add_argument('--train_sample_rate', type=float, default=0.1, help='Sampling rate for training datasets.')
    parser.add_argument('--test_sample_rate', type=float, default=1.0, help='Sampling rate for testing datasets.')
    parser.add_argument('--use_amp', type=int, default=1, help='Whether to use Automatic Mixed Precision.')
    
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 创建日志目录
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    # 创建日志文件，使用时间戳命名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(args.log_dir, f"{args.task}_{args.train_dataset}_to_{args.test_dataset}_{timestamp}.txt")
    
    # 打开日志文件
    log_file = open(log_filename, 'w')
    
    # 将标准输出重定向到文件和控制台
    class Tee(object):
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cpu")

    print(f"============= PID = {os.getpid()} ============= ")
    print(f"Log file created at: {log_filename}")
    print("Parameters:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    print(args)

    # 加载训练集
    train_dataset = performat_SramDataset(
        name=args.train_dataset, 
        dataset_dir='/local/hsl/datasets-dlpl', 
        neg_edge_ratio=0.3,
        to_undirected=args.to_undirected,
        sample_rates=args.train_sample_rate,
        task_type=args.task,
    )
    
    # 加载测试集
    test_dataset = performat_SramDataset(
        name=args.test_dataset, 
        dataset_dir='/local/hsl/datasets-dlpl', 
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
    print(f"运行完成，日志已保存到 {log_filename}")
