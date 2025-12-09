import types  
import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score,
    mean_absolute_error, mean_squared_error,
    mean_squared_error as root_mean_squared_error, r2_score,
)
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from model import CapClassifier, CapRegressor

import time
from tqdm import tqdm
from sampling import dataset_sampling
import numpy as np
from utils import get_model_params_dict, save_cap_model, load_cap_model, check_cap_model_exists

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
        # 检查是否为NumPy数组
        if isinstance(pred_score, np.ndarray):
            if len(pred_score.shape) == 1 or pred_score.shape[1] == 1:
                return (pred_score > 0.5).astype(np.int64)
            else:
                return np.argmax(pred_score, axis=1)
        else:
            # PyTorch张量
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
        reformat = lambda x: round(float(x), 6)

        if self.task == 'classification':
            pred_int = self._get_pred_int(pred_score)
            # 如果pred_int已经是numpy数组，就不需要再调用.numpy()
            if not isinstance(pred_int, np.ndarray):
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

        # 格式化输出结果
        result_str = f"{split.ljust(6)} | " + " | ".join([f"{k}: {v:.6f}" for k, v in res.items()])
        print(result_str)
        return res

class FocalLoss(torch.nn.Module):
    """
    Focal Loss实现，适用于多分类问题
    
    Args:
        gamma (float): 聚焦参数，控制难易样本的权重，默认为2.0
        reduction (str): 损失归约方式，'mean', 'sum' 或 'none'
        class_weights (Tensor): 各类别的权重，如果提供则自动计算alpha
    """
    def __init__(self, gamma=2.0, reduction='mean', class_weights=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights
        
    def forward(self, input, target):
        """
        Args:
            input: 模型输出的logits，形状为 [N, C]
            target: 目标类别索引，形状为 [N]
        """
        # 计算softmax概率
        log_probs = F.log_softmax(input, dim=1)
        probs = torch.exp(log_probs)
        
        # 获取目标类的概率
        target_probs = probs.gather(1, target.unsqueeze(1))
        
        # 计算focal loss的调制项
        focal_weight = (1 - target_probs) ** self.gamma
        
        # 应用类别权重（如果提供）
        if self.class_weights is not None:
            # 从target中获取每个样本对应的类别权重
            alpha = self.class_weights.gather(0, target)
            focal_weight = focal_weight * alpha.unsqueeze(1)
        
        # 计算带有focal weight的损失
        loss = -focal_weight * log_probs.gather(1, target.unsqueeze(1))
        
        # 归约损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
            
def compute_class_weights(labels, num_classes):
    """
    计算类别权重，权重为类别频率的倒数
    
    Args:
        labels: 类别标签
        num_classes: 类别总数
    
    Returns:
        class_weights: 每个类别的权重
    """
    # 计算每个类别的样本数
    counts = torch.zeros(num_classes, device=labels.device)
    for i in range(num_classes):
        counts[i] = (labels == i).sum().float()
    
    # 防止除零错误
    counts = counts.clamp(min=1.0)
    
    # 计算频率
    freqs = counts / counts.sum()
    
    # 权重为频率的倒数
    class_weights = 1.0 / freqs
    
    # 归一化权重使其总和为num_classes
    class_weights = class_weights * (num_classes / class_weights.sum())
    
    return class_weights

def compute_loss(pred, true, task, use_focal_loss=False):
    """Compute loss and prediction score. 
    Args:
        pred (torch.tensor): Unnormalized prediction
        true (torch.tensor): Groud truth label
        task (str): The task type, 'classification' or 'regression'
        use_focal_loss (bool): Whether to use focal loss for classification
    Returns: Loss, normalized prediction score
    """
    ## default manipulation for pred and true
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    if task == 'classification':
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

        ## multiclass
        if pred.ndim > 1 and true.ndim == 1:
            if use_focal_loss:
                # 计算类别权重
                num_classes = pred.size(1)
                class_weights = compute_class_weights(true, num_classes)
                
                # 应用Focal Loss
                focal_loss = FocalLoss(gamma=2.0, class_weights=class_weights)
                return focal_loss(pred, true), F.softmax(pred, dim=-1)
            else:
                # 默认使用交叉熵损失
                pred_log_softmax = F.log_softmax(pred, dim=-1)
                return F.nll_loss(pred_log_softmax, true), F.softmax(pred, dim=-1)
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
def eval_epoch(model, loader, device, split='val', task='classification', use_focal_loss=False):
    """ 
    Evaluate the classification model on a data loader
    """
    model.eval()
    time_start = time.time()
    logger = Logger(task=task)

    for batch in tqdm(loader, desc=f"eval_{split}", leave=False):
        batch = batch.to(device)
        
        # 前向传播
        y_pred, y = model(batch)
        loss, pred_score = compute_loss(y_pred, y, task, use_focal_loss=use_focal_loss)
        
        _true = y.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        
        logger.update_stats(
            true=_true,
            pred=_pred,
            batch_size=_true.squeeze().size(0),
            loss=loss.detach().cpu().item(),
        )
        
    print(f"\n----- {split.upper()} 评估 - 用时: {time.time() - time_start:.2f}s -----")
    return logger.write_epoch(split)

@torch.no_grad()
def eval_epoch_regress(classifier, regressor, loader, device, split='val', use_focal_loss=False):
    """
    Evaluate the regression model on a data loader
    """
    classifier.eval()
    regressor.eval()
    time_start = time.time()
    logger = Logger(task='regression')
    class_logger = Logger(task='classification')

    for batch in tqdm(loader, desc=f"eval_{split}", leave=False):
        batch = batch.to(device)

        # 前向传播 - 分类器
        class_logits, graph_embed = classifier(batch)
        predict_class = torch.argmax(class_logits, dim=1)
        true_class = classifier.classify_capacitance(batch.edge_label)
        
        # 计算分类损失
        if use_focal_loss:
            num_classes = class_logits.size(1)
            class_weights = compute_class_weights(true_class, num_classes)
            focal_loss = FocalLoss(gamma=2.0, class_weights=class_weights)
            class_loss = focal_loss(class_logits, true_class)
        else:
            class_loss = F.cross_entropy(class_logits, true_class)
        
        # 前向传播 - 回归器
        regressor_pred = regressor(graph_embed.detach(), predict_class)
        regressor_loss = F.mse_loss(regressor_pred.squeeze(), batch.edge_label.squeeze())

        # 收集统计信息 - 回归
        _pred = regressor_pred.detach().to('cpu', non_blocking=True)
        _true = batch.edge_label.detach().to('cpu', non_blocking=True)
        logger.update_stats(
            true=_true, 
            pred=_pred, 
            batch_size=_true.squeeze().size(0), 
            loss=regressor_loss.detach().cpu().item()
        )
        
        # 收集统计信息 - 分类
        _class_pred = predict_class.detach().to('cpu', non_blocking=True)
        _class_true = true_class.detach().to('cpu', non_blocking=True)
        class_logger.update_stats(
            true=_class_true, 
            pred=_class_pred, 
            batch_size=_class_true.squeeze().size(0), 
            loss=class_loss.detach().cpu().item()
        )
        
    print(f"\n----- {split.upper()} 评估 - 用时: {time.time() - time_start:.2f}s -----")
    reg_metrics = logger.write_epoch(split)
    print(f"分类结果:")
    class_metrics = class_logger.write_epoch(split)
    
    return reg_metrics, class_metrics

def train_classification_epoch(model, optimizer, loader, device, scaler, task='classification', use_focal_loss=False):
    """
    训练分类模型一个epoch，与原始class_train函数保持数据传递一致
    """
    model.train()
    logger = Logger(task=task)
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # 使用混合精度训练
        with amp.autocast():
            # 前向传播
            y_pred, y = model(batch)
            loss, pred = compute_loss(y_pred, y, task, use_focal_loss=use_focal_loss)
        
        # 保持与原始代码一致，使用y_pred而不是pred来统计
        _true = y.detach().to('cpu', non_blocking=True)
        _pred = y_pred.detach().to('cpu', non_blocking=True)
        
        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 收集统计信息 - 与原始代码一致
        logger.update_stats(
            true=_true, 
            pred=_pred, 
            batch_size=_true.squeeze().size(0), 
            loss=loss.detach().cpu().item()
        )
    
    return logger

def train_regress_epoch(classifier, regressor, optimizer_classifier, optimizer_regressor, 
                       loader, device, scaler, use_focal_loss=False):
    """
    训练回归模型一个epoch（包括分类器和回归器），与原始regress_train函数保持数据传递一致
    """
    classifier.train()
    regressor.train()
    logger = Logger(task='regression')
    class_logger = Logger(task='classification')
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        
        # 第1步: 训练分类器
        optimizer_classifier.zero_grad()
        
        with amp.autocast():
            # 分类器前向传播
            class_logits, graph_embed = classifier(batch)
            true_class = classifier.classify_capacitance(batch.edge_label)
            
            # 计算分类损失
            if use_focal_loss:
                num_classes = class_logits.size(1)
                class_weights = compute_class_weights(true_class, num_classes)
                focal_loss = FocalLoss(gamma=2.0, class_weights=class_weights)
                class_loss = focal_loss(class_logits, true_class)
            else:
                class_loss = F.cross_entropy(class_logits, true_class)
        
        # 分类器反向传播
        scaler.scale(class_loss).backward()
        scaler.step(optimizer_classifier)
        scaler.update()
        
        # 第2步: 训练回归器
        optimizer_regressor.zero_grad()
        
        # 与原始代码保持一致：先计算预测类别，然后使用detached的graph_embed和true_class
        predict_class = torch.argmax(class_logits, dim=1)
        
        with amp.autocast():
            # 回归器前向传播 - 使用true_class (与原始代码一致)
            regressor_pred = regressor(graph_embed.detach(), true_class)
            regressor_loss = F.mse_loss(regressor_pred.squeeze(), batch.edge_label.squeeze())
        
        # 回归器反向传播
        scaler.scale(regressor_loss).backward()
        scaler.step(optimizer_regressor)
        scaler.update()
        
        # 收集统计信息 - 回归 (与原始代码一致)
        _pred = regressor_pred.detach().to('cpu', non_blocking=True)
        _true = batch.edge_label.detach().to('cpu', non_blocking=True)
        logger.update_stats(
            true=_true, 
            pred=_pred, 
            batch_size=_true.squeeze().size(0), 
            loss=regressor_loss.detach().cpu().item()
        )
        
        # 收集统计信息 - 分类 (与原始代码一致)
        _class_pred = predict_class.detach().to('cpu', non_blocking=True)
        _class_true = true_class.detach().to('cpu', non_blocking=True)
        class_logger.update_stats(
            true=_class_true, 
            pred=_class_pred, 
            batch_size=_class_true.squeeze().size(0), 
            loss=class_loss.detach().cpu().item()
        )
    
    return logger, class_logger

def downstream_link_pred(args, dataset, device):
    """
    This function trains and evaluates the model.
    """
    print(f"====== Train {args.train_dataset} to Test {args.test_dataset} ======")
    
    # 获取训练集和测试集对象
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    if args.task == 'regression':
        ## normalize the circuit statistics
        train_dataset.norm_nfeat([NET, DEV])
        test_dataset.norm_nfeat([NET, DEV])
    
    # 使用dataset_sampling函数创建数据加载器
    print(f"调用dataset_sampling处理数据集...")
    train_loader, val_loader, test_loaders = dataset_sampling(args, train_dataset, test_dataset)
    print(f"数据加载器创建完成: 1个训练，1个验证，{len(test_loaders)}个测试")
    
    # 获取模型参数字典
    model_params = get_model_params_dict(args)
    
    # 检查是否存在预训练模型
    if args.task == 'classification':
        classifier_exists, classifier_path = check_cap_model_exists('classifier', f"{args.train_dataset}_to_{args.test_dataset}", model_params)
    else:  # regression
        classifier_exists, classifier_path = check_cap_model_exists('classifier', f"{args.train_dataset}_to_{args.test_dataset}", model_params)
        regressor_exists, regressor_path = check_cap_model_exists('regressor', f"{args.train_dataset}_to_{args.test_dataset}", model_params)

    # 初始化分类模型
    classifier = CapClassifier(
        hidden_dim=args.hid_dim, 
        num_classes=3 if args.task == 'regression' else 1,
        num_layers=args.num_gnn_layers, 
        use_bn=bool(args.use_bn),
        drop_out=args.dropout,
        activation=args.act_fn,
        src_dst_agg=args.src_dst_agg,
        task_type=args.task,
        use_stats=bool(args.use_stats)
    ).to(device)
    
    # 尝试加载预训练的分类器模型
    if classifier_exists:
        print(f"加载预训练分类器模型: {classifier_path}")
        load_cap_model(classifier, 'classifier', f"{args.train_dataset}_to_{args.test_dataset}", model_params)
    
    # 如果是回归任务，还需要初始化回归模型
    if args.task == 'regression':
        regressor = CapRegressor(
            hidden_dim=args.hid_dim, 
            num_classes=3,
            use_bn=bool(args.use_bn),
            drop_out=args.dropout,
            activation=args.act_fn,
            src_dst_agg=args.src_dst_agg
        ).to(device)
        
        # 尝试加载预训练的回归器模型
        if regressor_exists:
            print(f"加载预训练回归器模型: {regressor_path}")
            load_cap_model(regressor, 'regressor', f"{args.train_dataset}_to_{args.test_dataset}", model_params)

    # 优化器
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    
    if args.task == 'regression':
        regressor_optimizer = torch.optim.Adam(regressor.parameters(), lr=args.lr)
    
    # 创建混合精度训练的scaler
    scaler = amp.GradScaler(enabled=bool(args.use_amp))
    
    # 用于记录最佳性能
    best_metrics = {
        'acc': 0.0,
        'auc': 0.0,
        'mae': float('inf'),
        'r2': float('-inf'),
        'epoch': 0
    }
    
    # 训练循环
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        # ========== 训练阶段 ========== #
        if args.task == 'classification':
            # 分类任务
            train_logger = train_classification_epoch(
                classifier, classifier_optimizer, train_loader, device, 
                scaler, task=args.task, use_focal_loss=args.use_focal_loss
            )
            
            print(f"\n===== Epoch {epoch}/{args.epochs} - Elapsed: {time.time() - epoch_start_time:.2f}s =====")
            train_metrics = train_logger.write_epoch(split='train')
            
            # 与原始代码保持一致：每个epoch都评估
            # ========== 验证阶段 ========== #
            val_metrics = eval_epoch(
                classifier, val_loader, device, 
                split='val', task=args.task, use_focal_loss=args.use_focal_loss
            )
            
            # ========== 测试阶段 ========== #
            test_results = {}
            for test_name, test_loader in test_loaders.items():
                test_metrics = eval_epoch(
                    classifier, test_loader, device, 
                    split=f'test_{test_name}', task=args.task, use_focal_loss=args.use_focal_loss
                )
                test_results[test_name] = test_metrics
            
            # 更新最佳性能
            is_best = val_metrics['auc'] > best_metrics['auc']
            if is_best:
                best_metrics['acc'] = val_metrics['accuracy']
                best_metrics['auc'] = val_metrics['auc']
                best_metrics['epoch'] = epoch
                
                # 保存最佳模型
                metrics = {
                    'acc': float(val_metrics['accuracy']),
                    'auc': float(val_metrics['auc']),
                    'loss': float(val_metrics['loss']),
                    'test_results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in test_results.items()}
                }
                
                save_cap_model(classifier, 'classifier', f"{args.train_dataset}_to_{args.test_dataset}", 
                             epoch, model_params, is_best, metrics)
            
        else:  # 回归任务
            # 回归任务
            train_logger, train_class_logger = train_regress_epoch(
                classifier, regressor, classifier_optimizer, regressor_optimizer,
                train_loader, device, scaler, use_focal_loss=args.use_focal_loss
            )
            
            print(f"\n===== Epoch {epoch}/{args.epochs} - Elapsed: {time.time() - epoch_start_time:.2f}s =====")
            train_metrics = train_logger.write_epoch(split='train')
            print("分类结果:")
            train_class_metrics = train_class_logger.write_epoch(split='train')
            
            # 与原始代码保持一致：每个epoch都评估
            # ========== 验证阶段 ========== #
            val_metrics, val_class_metrics = eval_epoch_regress(
                classifier, regressor, val_loader, device, 
                split='val', use_focal_loss=args.use_focal_loss
            )
            
            # ========== 测试阶段 ========== #
            test_results = {}
            for test_name, test_loader in test_loaders.items():
                test_metrics, test_class_metrics = eval_epoch_regress(
                    classifier, regressor, test_loader, device, 
                    split=f'test_{test_name}', use_focal_loss=args.use_focal_loss
                )
                test_results[test_name] = test_metrics
            
            # 更新最佳性能 - 分类器
            is_best_classifier = val_class_metrics['accuracy'] > best_metrics['acc']
            if is_best_classifier:
                best_metrics['acc'] = val_class_metrics['accuracy']
                
                # 保存最佳分类器模型
                metrics = {
                    'acc': float(val_class_metrics['accuracy']),
                    'loss': float(val_class_metrics['loss']),
                    'test_results': {k: {'acc': v['accuracy']} for k, v in test_results.items()}
                }
                
                save_cap_model(classifier, 'classifier', f"{args.train_dataset}_to_{args.test_dataset}", 
                             epoch, model_params, is_best_classifier, metrics)
            
            # 更新最佳性能 - 回归器
            is_best_regressor = val_metrics['r2'] > best_metrics['r2']
            if is_best_regressor:
                best_metrics['mae'] = val_metrics['mae']
                best_metrics['r2'] = val_metrics['r2']
                best_metrics['epoch'] = epoch
                
                # 保存最佳回归器模型
                metrics = {
                    'mae': float(val_metrics['mae']),
                    'r2': float(val_metrics['r2']),
                    'loss': float(val_metrics['loss']),
                    'test_results': {k: {'mae': v['mae'], 'r2': v['r2']} for k, v in test_results.items()}
                }
                
                save_cap_model(regressor, 'regressor', f"{args.train_dataset}_to_{args.test_dataset}", 
                             epoch, model_params, is_best_regressor, metrics)
        
        print("=" * 50)
    
    # 打印最终结果
    if args.task == 'classification':
        print(f"最佳性能 (Epoch {best_metrics['epoch']}): Val Acc: {best_metrics['acc']:.4f}, Val AUC: {best_metrics['auc']:.4f}")
    else:
        print(f"最佳性能 (Epoch {best_metrics['epoch']}): Val MAE: {best_metrics['mae']:.4f}, Val R2: {best_metrics['r2']:.4f}")
    
    # 在所有测试集上进行最终评估
    print("\n===== 最终测试结果 =====")
    for test_name, test_loader in test_loaders.items():
        if args.task == 'classification':
            test_metrics = eval_epoch(
                classifier, test_loader, device, 
                split=f'test_{test_name}', task=args.task, use_focal_loss=args.use_focal_loss
            )
            print(f"测试集 ({test_name}): Acc = {test_metrics['accuracy']:.4f}, AUC = {test_metrics['auc']:.4f}")
        else:
            test_metrics, _ = eval_epoch_regress(
                classifier, regressor, test_loader, device, 
                split=f'test_{test_name}', use_focal_loss=args.use_focal_loss
            )
            print(f"测试集 ({test_name}): MAE = {test_metrics['mae']:.4f}, R2 = {test_metrics['r2']:.4f}")