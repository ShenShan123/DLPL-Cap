import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score,
    mean_absolute_error, mean_squared_error,
    root_mean_squared_error, r2_score,
)

# from torch.utils.data.sampler import SubsetRandomSampler
# from sram_dataset import LinkPredictionDataset
# from sram_dataset import collate_fn, adaption_for_sgrl
# from torch_geometric.data import Batch

import time
import os
import sys
sys.path.append('..')
from utils_model_checkpoint import check_model_exists, save_model, load_model, get_model_params_dict
from tqdm import tqdm
# from torch_geometric.loader import NeighborLoader, GraphSAINTRandomWalkSampler, GraphSAINTEdgeSampler, ShaDowKHopSampler
from model import GraphHead
from sampling_and_pe import dataset_sampling_and_pe_calculation

NET = 0
DEV = 1
PIN = 2

class Logger (object):
    """ 
    Logger for printing message during training and evaluation. 
    Adapted from GraphGPS 
    """
    
    def __init__(self, task='classification'):
        super().__init__()
        # Whether to run comparison tests of alternative score implementations.
        self.test_scores = False
        self._iter = 0
        self._true = []
        self._pred = []
        self._loss = 0.0
        self._size_current = 0
        self.task = task

    def _get_pred_int(self, pred_score):
        if len(pred_score.shape) == 1 or pred_score.shape[1] == 1:
            return (pred_score > 0.5).astype(int)
        else:
            return pred_score.max(dim=1)[1]

    def update_stats(self, true, pred, batch_size, loss):
        self._true.append(true)
        self._pred.append(pred)
        self._size_current += batch_size
        self._loss += loss * batch_size
        self._iter += 1

    def write_epoch(self, split=""):
        true, pred_score = torch.cat(self._true), torch.cat(self._pred)
        true = true.numpy()
        pred_score = pred_score.numpy()
        reformat = lambda x: round(float(x), 4)

        if self.task == 'classification':
            pred_int = self._get_pred_int(pred_score)

            try:
                r_a_score = roc_auc_score(true, pred_score)
            except ValueError:
                r_a_score = 0.0

            # performance metrics to be printed
            res = {
                'loss': round(self._loss / self._size_current, 8),
                'accuracy': reformat(accuracy_score(true, pred_int)),
                'precision': reformat(precision_score(true, pred_int)),
                'recall': reformat(recall_score(true, pred_int)),
                'f1': reformat(f1_score(true, pred_int)),
                'auc': reformat(r_a_score),
            }
        else:
            res = {
                'loss': round(self._loss / self._size_current, 8),
                'mae': reformat(mean_absolute_error(true, pred_score)),
                'mse': reformat(mean_squared_error(true, pred_score)),
                'rmse': reformat(root_mean_squared_error(true, pred_score)),
                'r2': reformat(r2_score(true, pred_score)),
            }

        # 结果打印到屏幕
        print(split, res)
        return res

def compute_loss(args, pred, true, criterion):
    """Compute loss and prediction score. 
    This version only supports binary classification.
    Args:
        args (argparse.Namespace): The arguments
        pred (torch.tensor): Unnormalized prediction
        true (torch.tensor): Groud truth label
        criterion (torch.nn.Module): The loss function
    Returns: Loss, normalized prediction score
    """
    assert criterion, "Loss function is not provided!"
    ## default manipulation for pred and true
    ## can be skipped if special loss computation is needed
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    if args.task == 'classification':
        ## multiclass task uses the negative log likelihood loss.
        if pred.ndim > 1 and true.ndim == 1:
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true), pred
        ## binary or multilabel
        else:
            true = true.float()
            return criterion(pred, true), torch.sigmoid(pred)
        
    elif args.task == 'regression':
        # true = true.float()
        return criterion(pred, true), pred
    
    else:
        raise ValueError(f"Task type {args.task} not supported!")

@torch.no_grad()
def eval_epoch(args, loader, batched_dspd, model, device, 
               split='val', criterion=None):
    """ 
    evaluate the model on the validation or test set
    Args:
        args (argparse.Namespace): The arguments
        loader (torch.utils.data.DataLoader): The data loader
        model (torch.nn.Module): The model
        device (torch.device): The device to run the model on
        split (str): The split name, 'val' or 'test'
    """
    model.eval()
    time_start = time.time()
    logger = Logger(task=args.task)

    for i, batch in enumerate(tqdm(loader, desc="eval_"+split, leave=False)):
        ## copy dspd tensor to the batch
        batch.dspd = batched_dspd[i]
        pred, true = model(batch.to(device))
        loss, pred_score = compute_loss(args, pred, true, criterion=criterion)
        _true = true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            batch_size=_true.size(0),
                            loss=loss.detach().cpu().item(),
                            )
    return logger.write_epoch(split)

def train(args, model, optimizier, criterion,
          train_loader, val_loader, test_loaders, 
          train_batched_dspd, val_batched_dspd, 
          test_batched_dspd_dict, device):
    """
    Train the head model for link prediction task
    Args:
        args (argparse.Namespace): The arguments
        head_model (torch.nn.Module): The head model
        optimizier (torch.optim.Optimizer): The optimizer
        criterion (torch.nn.Module): The loss function
        train_loader (torch.utils.data.DataLoader): The training data loader
        val_loader (torch.utils.data.DataLoader): The validation data loader  
        test_laders (list): A list of test data loaders
        train_batched_dspd (list): The list of batched DSPD tensors for training
        val_batched_dspd (list): The list of batched DSPD tensors for validation
        test_batched_dspd_dict (dict): The dictionary of batched DSPD tensors for test datasets
        device (torch.device): The device to train the model on
    """
    # 导入内存监控工具
    import psutil
    import gc
    
    # 内存监控函数
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
        return memory_gb
        
    # 获取数据集名称
    dataset_name = args.train_dataset.split('+')[0] if '+' in args.train_dataset else args.train_dataset
    
    # 获取模型参数字典，用于唯一标识模型
    model_params = get_model_params_dict(args, 
                                        exclude_keys=['num_workers', 'gpu', 'epochs', 'batch_size', 'seed'])
    
    # 检查是否存在已训练的模型
    model_exists, model_path = check_model_exists("Cirgps", dataset_name, model_params)
    if model_exists:
        # 尝试加载已有模型
        print(f"发现已有Cirgps模型: {model_path}")
        try:
            load_success = load_model(model, "Cirgps", dataset_name, model_params)
            if load_success:
                print("成功加载Cirgps模型。评估模型性能...")
                # 评估加载的模型
                val_results = eval_epoch(args, val_loader, val_batched_dspd, model, device, 
                                       split='val', criterion=criterion)
                
                # 在测试集上评估
                test_results = []
                test_names = []
                for test_name, (test_loader, test_batched_dspd) in test_loaders.items():
                    test_result = eval_epoch(args, test_loader, test_batched_dspd, model, device, 
                                          split=f'test_{test_name}', criterion=criterion)
                    test_results.append(test_result)
                    test_names.append(test_name)
                
                # 创建结果文件
                result_file = open('test_results_loaded_model.txt', 'w')
                result_file.write(f"加载的预训练模型: {model_path}\n")
                result_file.write(f"验证集结果: {val_results}\n\n")
                for name, result in zip(test_names, test_results):
                    result_file.write(f"测试集 {name} 结果: {result}\n")
                result_file.close()
                
                print("已加载预训练模型，可以选择跳过训练过程")
                # 这里可以添加逻辑，根据需要决定是否继续训练
                # 如果训练时间重要，可以直接返回
                # return val_results, test_results, test_names
        except Exception as e:
            print(f"加载模型失败: {e}，将继续训练新模型")
    
    optimizier.zero_grad()
    
    best_results = {
        'best_val_mse': 1e9, 'best_val_loss': 1e9, 
        'best_epoch': 0, 'test_results': [], 'test_names': []
    }
    
    # 创建结果文件
    result_file = open('test_results.txt', 'w')
    result_file.write(f"训练参数 u: {args.u}\n")
    result_file.write(f"训练数据集: {args.train_dataset}, 采样率: {args.train_sample_rate}\n")
    result_file.write(f"测试数据集: {args.test_dataset}, 采样率: {args.test_sample_rate}\n\n")
    
    # 检查初始内存
    print("训练开始前内存使用情况:")
    initial_mem = print_memory_usage()
    
    # 设置内存检查阈值
    mem_warning_threshold = 0.9  # 90%的初始内存使用时发出警告
    mem_critical_threshold = 2.0  # 2倍初始内存使用时主动减少内存使用
    
    for epoch in range(args.epochs):
        # 检查训练前的内存使用
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        print("训练前内存状态:")
        current_mem = print_memory_usage()
        
        # 如果内存使用超过警告阈值，进行垃圾回收
        if current_mem > initial_mem * mem_warning_threshold:
            print(f"内存使用增加到 {current_mem:.2f} GB，执行垃圾回收...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 检查是否超过临界阈值
            if current_mem > initial_mem * mem_critical_threshold:
                print("内存使用超过临界值，尝试减少不必要的数据...")
                # 在极端情况下，可以考虑减小训练数据或减少缓存
                
        logger = Logger(task=args.task)
        model.train()

        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch:{epoch}')):
            optimizier.zero_grad()
            ## copy dspd tensor to the data batch
            batch.dspd = train_batched_dspd[i]
            
            ## Get the prediction from the model
            y_pred, y = model(batch.to(device))
            loss, pred = compute_loss(args, y_pred, y, criterion=criterion)
            _true = y.detach().to('cpu', non_blocking=True)
            _pred = y_pred.detach().to('cpu', non_blocking=True)
            
            loss.backward()
            optimizier.step()
            
            logger.update_stats(true=_true,
                                pred=pred.detach().to('cpu', non_blocking=True), 
                                batch_size=_true.size(0),
                                loss=loss.detach().cpu().item(),
                               )
                               
            # 每100个批次检查一次内存使用情况
            if i > 0 and i % 100 == 0:
                # 检查内存使用
                current_mem = print_memory_usage()
                if current_mem > initial_mem * mem_critical_threshold:
                    print(f"批次 {i}: 内存使用过高，执行垃圾回收...")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
        ## Get train results this epoch
        print("Train results this epoch")
        train_results = logger.write_epoch()

        # 验证前清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # validate
        print("Validate after epoch")
        val_results = eval_epoch(args, val_loader, val_batched_dspd, model, device, 
                               split='val', criterion=criterion)
        
        # 保存当前训练指标
        metrics = {
            "train_loss": train_results["loss"],
            "val_loss": val_results["loss"]
        }
        
        if args.task == 'classification':
            metrics.update({
                "train_accuracy": train_results["accuracy"],
                "val_accuracy": val_results["accuracy"],
                "train_auc": train_results["auc"],
                "val_auc": val_results["auc"]
            })
            is_best = val_results["auc"] > best_results.get("best_val_auc", 0)
            if is_best:
                best_results["best_val_auc"] = val_results["auc"]
                best_results["best_epoch"] = epoch
        else:  # regression
            metrics.update({
                "train_mse": train_results["mse"],
                "val_mse": val_results["mse"],
                "train_r2": train_results["r2"],
                "val_r2": val_results["r2"]
            })
            is_best = val_results["mse"] < best_results["best_val_mse"]
            if is_best:
                best_results["best_val_mse"] = val_results["mse"]
                best_results["best_epoch"] = epoch
                
        # 保存模型
        save_model(
            model=model,
            model_name="Cirgps",
            dataset_name=dataset_name,
            epoch=epoch,
            params=model_params,
            is_best=is_best,
            metrics=metrics
        )
        
        # Test when getting best val results
        if is_best:
            # 测试前清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            result_file.write(f"Epoch {epoch}, 验证集结果: {val_results}\n")
            test_results = []
            test_names = []
            for test_name, (test_loader, test_batched_dspd) in test_loaders.items():
                test_result = eval_epoch(args, test_loader, test_batched_dspd, model, device, 
                                      split=f'test_{test_name}', criterion=criterion)
                test_results.append(test_result)
                test_names.append(test_name)
                result_file.write(f"测试集 {test_name} 结果: {test_result}\n")
            result_file.write("\n")
            
            best_results["test_results"] = test_results
            best_results["test_names"] = test_names
            
        # Epoch结束后强制进行垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    result_file.close()
    return best_results

def downstream_train(args, dataset, device):
    """ downstream task training for link prediction
    Args:
        args (argparse.Namespace): The arguments
        dataset (dict): Dictionary containing 'train' and 'test' datasets
        device (torch.device): The device to train the model on
    """
    model = GraphHead(args)

    
    ## Subgraph sampling for each dataset graph & PE calculation
    (
        train_loader, val_loader, test_loaders,
        train_dspd_list, valid_dspd_list, test_dspd_dict,
    ) = dataset_sampling_and_pe_calculation(args, dataset['train'], dataset['test'])

    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    optimizier = torch.optim.Adam(model.parameters(),lr=args.lr)

    if args.task == 'classification':
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        print(f"Task is {args.task}, using BCEWithLogitsLoss")

    else:
        criterion = torch.nn.MSELoss(reduction='mean')
    
    start = time.time()

    ## Start training, go go go!
    train(args, model, optimizier, criterion,
          train_loader, val_loader, test_loaders, 
          train_dspd_list, valid_dspd_list, 
          test_dspd_dict, device)
    
    elapsed = time.time() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    print(f"Done! Training took {timestr}")