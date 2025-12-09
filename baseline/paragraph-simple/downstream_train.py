import types  
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score,
    mean_absolute_error, mean_squared_error,
    mean_squared_error as root_mean_squared_error, r2_score,
)
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data

import time
import os
import sys
sys.path.append('..')
from utils_model_checkpoint import check_model_exists, save_model, load_model, get_model_params_dict
from tqdm import tqdm
# from torch_geometric.loader import NeighborLoader, GraphSAINTRandomWalkSampler, GraphSAINTEdgeSampler, ShaDowKHopSampler
from model import GraphHead
from Ensemble_model import EnsembleModel
from sampling import dataset_sampling



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
            return (pred_score > 0.5).long()
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
        true = true.cpu().numpy()
        pred_score = pred_score.cpu().numpy()
        reformat = lambda x: round(float(x), 4)

        if self.task == 'classification':
            pred_int = self._get_pred_int(pred_score)
            pred_int = pred_int.numpy()

            try:
                r_a_score = roc_auc_score(true, pred_score)
            except ValueError:
                r_a_score = 0.0

            # performance metrics to be printed
            res = {
                'loss': reformat(self._loss / self._size_current),
                'accuracy': reformat(accuracy_score(true, pred_int)),
                'precision': reformat(precision_score(true, pred_int)),
                'recall': reformat(recall_score(true, pred_int)),
                'f1': reformat(f1_score(true, pred_int)),
                'auc': reformat(r_a_score),
            }
        else:
            res = {
                'loss': reformat(self._loss / self._size_current),
                'mae': reformat(mean_absolute_error(true, pred_score)),
                'mse': reformat(mean_squared_error(true, pred_score)),
                'rmse': reformat(root_mean_squared_error(true, pred_score)),
                'r2': reformat(r2_score(true, pred_score)),
            }

        # Just print the results to screen
        print(split, res)
        return res

def compute_loss(pred, true, task):
    """Compute loss and prediction score. 
    This version only supports binary classification.
    Args:
        pred (torch.tensor): Unnormalized prediction
        true (torch.tensor): Groud truth label
        task (str): The task type, 'classification' or 'regression'
    Returns: Loss, normalized prediction score
    """

    ## default manipulation for pred and true
    ## can be skipped if special loss computation is needed
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    if task == 'classification':
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

        ## multiclass
        if pred.ndim > 1 and true.ndim == 1:
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true), pred
        ## binary or multilabel
        else:
            true = true.float()
            return bce_loss(pred, true), torch.sigmoid(pred)
        
    elif task == 'regression':
        mse_loss = torch.nn.MSELoss(reduction='mean')
        return mse_loss(pred, true), pred
    
    else:
        raise ValueError(f"Task type {task} not supported!")

@torch.no_grad()
def eval_epoch(loader, model, device, 
               split='val', task='classification'):
    """ 
    evaluate the model on the validation or test set
    Args:
        loader (torch.utils.data.DataLoader): The data loader
        model (torch.nn.Module): The model
        device (torch.device): The device to run the model on
        split (str): The split name, 'val' or 'test'
        task (str): The edge-level task type, 'classification' or 'regression'
    """
    model.eval()
    time_start = time.time()
    logger = Logger(task=task)

    for i, batch in enumerate(tqdm(loader, desc="eval_"+split, leave=False)):
        ## copy dspd tensor to the batch
        pred, true = model(batch.to(device))
        loss, pred_score = compute_loss(pred, true, task)
        _true = true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            batch_size=_true.squeeze().size(0),
                            loss=loss.detach().cpu().item(),
                            )
    logger.write_epoch(split)

def train(args, model, optimizers, 
          train_loader, val_loader, test_loaders, 
          device):
    """
    Train the head model for link prediction task
    Args:
        args (argparse.Namespace): The arguments
        model (torch.nn.Module): The model
        optimizers (list or torch.optim.Optimizer): The optimizer(s)
        train_loader (torch.utils.data.DataLoader): The training data loader
        val_loader (torch.utils.data.DataLoader): The validation data loader  
        test_laders (list): A list of test data loaders
        device (torch.device): The device to train the model on
    """
    # 获取数据集名称
    dataset_name = args.train_dataset.split('+')[0] if '+' in args.train_dataset else args.train_dataset
    
    # 获取模型参数字典，用于唯一标识模型
    model_params = get_model_params_dict(args, 
                                        exclude_keys=['num_workers', 'gpu', 'epochs', 'batch_size', 'seed'])
    
    # 检查是否存在已训练的模型
    model_exists, model_path = check_model_exists("paragraph-simple", dataset_name, model_params)
    if model_exists:
        # 尝试加载已有模型
        print(f"发现已有paragraph-simple模型: {model_path}")
        try:
            load_success = load_model(model, "paragraph-simple", dataset_name, model_params)
            if load_success:
                print("成功加载paragraph-simple模型。评估模型性能...")
                # 评估加载的模型
                val_result = eval_epoch(val_loader, model, device, split='val', task=args.task)
                
                # 在测试集上评估
                for test_name, test_loader in test_loaders.items():
                    test_result = eval_epoch(test_loader, model, device, 
                                           split=f'test_{test_name}', task=args.task)
                
                print("已加载预训练模型，可以选择跳过训练过程")
                # 这里可以选择返回或继续训练
                # return
        except Exception as e:
            print(f"加载模型失败: {e}，将继续训练新模型")
    
    # Reset optimizers
    if isinstance(optimizers, list):
        for opt in optimizers:
            opt.zero_grad()
    else:
        optimizers.zero_grad()
    
    # 创建日志文件
    log_dir = f"./logs/{dataset_name}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(log_file, 'w') as f:
        f.write(f"Paragraph-Simple 训练日志\n")
        f.write(f"数据集: {dataset_name}\n")
        f.write(f"训练参数: {model_params}\n\n")
        
        best_val_metrics = {"loss": float('inf'), "auc": 0} if args.task == 'classification' else {"loss": float('inf'), "r2": -float('inf')}
        best_epoch = 0
        
        for epoch in range(args.epochs):
            logger = Logger(task=args.task)
            model.train()
    
            for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch:{epoch}')):
                # Move batch to device
                batch = batch.to(device)
                
                if args.use_ensemble and isinstance(model, EnsembleModel):
                    # Custom ensemble training step
                    loss, _pred, _true = model.train_step(batch, optimizers)
                else:
                    # Standard single model training step
                    if isinstance(optimizers, list):
                        for opt in optimizers:
                            opt.zero_grad()
                    else:
                        optimizers.zero_grad()
                    
                    ## Get the prediction from the model
                    y_pred, y = model(batch)
                    # print("y_pred", y_pred.squeeze()[:10])
                    # print("y_true", y.squeeze()[:10])
                    loss, pred = compute_loss(y_pred, y, args.task)
                    _true = y.detach().to('cpu', non_blocking=True)
                    _pred = pred.detach().to('cpu', non_blocking=True)
    
                    loss.backward()
                    
                    if isinstance(optimizers, list):
                        for opt in optimizers:
                            opt.step()
                    else:
                        optimizers.step()
                
                logger.update_stats(true=_true,
                                    pred=_pred, 
                                    batch_size=_true.size(0),
                                    loss=loss.detach().cpu().item(),
                                   )
                
            # 获取训练结果
            train_results = logger.write_epoch("Train")
            f.write(f"Epoch {epoch+1}/{args.epochs} - 训练: ")
            for k, v in train_results.items():
                f.write(f"{k}={v:.6f} ")
            f.write("\n")
            
            # 验证
            val_logger = Logger(task=args.task)
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    y_pred, y = model(batch)
                    loss, pred = compute_loss(y_pred, y, args.task)
                    
                    val_logger.update_stats(
                        true=y.detach().to('cpu', non_blocking=True),
                        pred=pred.detach().to('cpu', non_blocking=True),
                        batch_size=y.size(0),
                        loss=loss.detach().cpu().item()
                    )
            
            val_results = val_logger.write_epoch("Val")
            f.write(f"Epoch {epoch+1}/{args.epochs} - 验证: ")
            for k, v in val_results.items():
                f.write(f"{k}={v:.6f} ")
            f.write("\n")
            
            # 保存训练指标
            metrics = {
                "train_loss": train_results["loss"],
                "val_loss": val_results["loss"]
            }
            
            # 判断当前模型是否是最佳模型
            is_best = False
            if args.task == 'classification':
                # 分类任务使用AUC作为指标
                metrics.update({
                    "train_auc": train_results["auc"],
                    "val_auc": val_results["auc"]
                })
                if val_results["auc"] > best_val_metrics["auc"]:
                    is_best = True
                    best_val_metrics["auc"] = val_results["auc"]
                    best_val_metrics["loss"] = val_results["loss"]
                    best_epoch = epoch
            else:
                # 回归任务使用R2作为指标
                metrics.update({
                    "train_r2": train_results["r2"],
                    "val_r2": val_results["r2"]
                })
                if val_results["r2"] > best_val_metrics["r2"]:
                    is_best = True
                    best_val_metrics["r2"] = val_results["r2"]
                    best_val_metrics["loss"] = val_results["loss"]
                    best_epoch = epoch
            
            # 保存模型
            save_model(
                model=model,
                model_name="paragraph-simple",
                dataset_name=dataset_name,
                epoch=epoch,
                params=model_params,
                is_best=is_best,
                metrics=metrics
            )
            
            # 如果是最佳模型，在测试集上评估
            if is_best:
                f.write(f"【新的最佳模型！】Epoch {epoch+1}\n")
                for test_name, test_loader in test_loaders.items():
                    test_logger = Logger(task=args.task)
                    model.eval()
                    with torch.no_grad():
                        for batch in test_loader:
                            batch = batch.to(device)
                            y_pred, y = model(batch)
                            loss, pred = compute_loss(y_pred, y, args.task)
                            
                            test_logger.update_stats(
                                true=y.detach().to('cpu', non_blocking=True),
                                pred=pred.detach().to('cpu', non_blocking=True),
                                batch_size=y.size(0),
                                loss=loss.detach().cpu().item()
                            )
                    
                    test_results = test_logger.write_epoch(f"Test_{test_name}")
                    f.write(f"测试集 {test_name}: ")
                    for k, v in test_results.items():
                        f.write(f"{k}={v:.6f} ")
                    f.write("\n")
                
                f.write("\n")
        
        # 训练结束，记录最终结果
        f.write(f"\n训练完成!\n")
        if args.task == 'classification':
            f.write(f"最佳模型在第 {best_epoch+1} 轮, 验证AUC = {best_val_metrics['auc']:.6f}, 验证损失 = {best_val_metrics['loss']:.6f}\n")
        else:
            f.write(f"最佳模型在第 {best_epoch+1} 轮, 验证R2 = {best_val_metrics['r2']:.6f}, 验证损失 = {best_val_metrics['loss']:.6f}\n")
    
    print(f"训练完成! 最佳模型在第 {best_epoch+1} 轮")
    print(f"训练日志已保存到 {log_file}")
    
    # 确保最终加载的是最佳模型
    try:
        best_model_path = os.path.join(f"./models_paragraph-simple", dataset_name, "paragraph-simple_best.pth")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print(f"已加载最佳模型: {best_model_path}")
    except Exception as e:
        print(f"加载最佳模型失败: {e}")
    
    return model

NET = 0
DEV = 1
PIN = 2

def downstream_link_pred(args, dataset, device):
    """ downstream task training for link prediction
    Args:
        args (argparse.Namespace): The arguments
        dataset (dict): Dictionary containing 'train' and 'test' datasets
        device (torch.device): The device to train the model on
    """
    # 对训练和测试数据集分别进行处理
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    if args.task == 'regression':
        ## 规范化回归任务的特征
        train_dataset.norm_nfeat([0, 1])  # 0=NET, 1=DEV
        test_dataset.norm_nfeat([0, 1])
    
    ## 子图采样
    (
        train_loader, val_loader, test_loaders
    ) = dataset_sampling(args, dataset)
    
    # Create model based on whether ensemble is enabled
    if args.use_ensemble:
        print(f"Creating ensemble model with {args.num_ensemble} sub-models")
        model = EnsembleModel(args, device, thresholds=args.ensemble_thresholds)
        
        # Create optimizers for each model in the ensemble
        optimizers = [
            torch.optim.Adam(model.models[i].parameters(), lr=args.lr) 
            for i in range(len(model.models))
        ]
    else:
        # Single model (original implementation)
        model = GraphHead(
            args.hid_dim, 1, num_layers=args.num_gnn_layers, 
            num_head_layers=args.num_head_layers, 
            use_bn=args.use_bn, drop_out=args.dropout, activation=args.act_fn, 
            src_dst_agg=args.src_dst_agg, max_dist=args.max_dist,
            task=args.task
        )
        model = model.to(device)
        optimizers = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    start = time.time()

    ## Start training, go go go!
    train(args, model, optimizers, 
          train_loader, val_loader, test_loaders, 
          device)

    # Move model back to original device
    model.to(device)
    elapsed = time.time() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed))

    print(f"Done! Training took {timestr}")