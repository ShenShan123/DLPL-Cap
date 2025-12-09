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
from tqdm import tqdm
# from torch_geometric.loader import NeighborLoader, GraphSAINTRandomWalkSampler, GraphSAINTEdgeSampler, ShaDowKHopSampler
from model import GraphHead
from sampling_and_pe import dataset_sampling_and_pe_calculation
from balanced_mse import GAILoss, BMCLoss, train_gmm

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

        # Just print the results to screen
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
    optimizier.zero_grad()
    
    if args.task == 'regression':
        best_results = {
            'best_val_mse': 1e9, 'best_val_loss': 1e9, 
            'best_epoch': 0, 'test_results': []
        }
    elif args.task == 'classification':
        best_results = {
            'best_val_f1': 0, 'best_val_loss': 1e9, 
            'best_epoch': 0, 'test_results': []
        }
    
    for epoch in range(args.epochs):
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
            
            ## Update the logger and print message to the screen
            logger.update_stats(
                true=_true, pred=_pred, 
                batch_size=_true.squeeze().size(0), 
                loss=loss.detach().cpu().item()
            )

        logger.write_epoch(split='train')
        ## ========== validation ========== ##
        val_res = eval_epoch(
            args, val_loader, val_batched_dspd, 
            model, device, split='val', criterion=criterion
        )

        ## update the best results so far
        if args.task == 'regression':
            if best_results['best_val_mse'] > val_res['mse']:
                best_results['best_val_mse'] = val_res['mse']
                best_results['best_val_loss'] = val_res['loss']
                best_results['best_epoch'] = epoch
        elif args.task == 'classification':
            if best_results['best_val_f1'] < val_res['f1']:
                best_results['best_val_f1'] = val_res['f1']
                best_results['best_val_loss'] = val_res['loss']
                best_results['best_epoch'] = epoch
        
        test_results = []

        ## ========== testing on other datasets ========== ##
        for test_name in test_batched_dspd_dict.keys():
            res = eval_epoch(
                args, test_loaders[test_name], 
                test_batched_dspd_dict[test_name], 
                model, device, split='test', 
                criterion=criterion
            )
            test_results.append(res)

        if best_results['best_epoch'] == epoch:
            best_results['test_results'] = test_results
        
        if args.task == 'regression':
            print( "=====================================")
            print(f" Best epoch: {best_results['best_epoch']}, mse: {best_results['best_val_mse']}, loss: {best_results['best_val_loss']}")
            print(f" Test results: {[res for res in best_results['test_results']]}")
            print( "=====================================")
        elif args.task == 'classification':
            print( "=====================================")
            print(f" Best epoch: {best_results['best_epoch']}, f1: {best_results['best_val_f1']}, loss: {best_results['best_val_loss']}")
            print(f" Test results: {[res for res in best_results['test_results']]}")
            print( "=====================================")


def downstream_train(args, dataset, device, subgraph_embeddings=None, edge_id_mapping=None,
                  train_loader=None, val_loader=None, test_loaders=None,
                  train_dspd_list=None, valid_dspd_list=None, test_dspd_dict=None):
    """ downstream task training for link prediction
    Args:
        args (argparse.Namespace): The arguments
        dataset (torch_geometric.data.InMemoryDataset): The dataset
        device (torch.device): The device to train the model on
        subgraph_embeddings (dict): 子图嵌入字典，由GraphCL生成，键为边ID，值为嵌入向量
        edge_id_mapping (dict): 边ID到嵌入索引的映射
        train_loader (torch_geometric.loader.LinkNeighborLoader): 训练数据加载器
        val_loader (torch_geometric.loader.LinkNeighborLoader): 验证数据加载器
        test_loaders (dict): 测试数据加载器字典
        train_dspd_list (list): 训练数据的距离编码列表
        valid_dspd_list (list): 验证数据的距离编码列表
        test_dspd_dict (dict): 测试数据的距离编码字典
    """
    model = GraphHead(args)
    model = model.to(device)


    
    # 如果没有传入加载器和距离编码，则进行采样
    if train_loader is None or val_loader is None or test_loaders is None:
        (
            train_loader, val_loader, test_loaders,
            train_dspd_list, valid_dspd_list, test_dspd_dict,
        ) = dataset_sampling_and_pe_calculation(args, dataset['train'], dataset['test'])
    
    # 如果有子图嵌入，将其添加到模型中
    if subgraph_embeddings is not None and edge_id_mapping is not None:
        model.set_subgraph_embeddings(subgraph_embeddings, edge_id_mapping)
        print(f"Added {len(subgraph_embeddings)} GraphCL subgraph embeddings to model")
    elif args.sgrl:
        # 兼容旧的节点级嵌入方式
        print("Using traditional node embeddings from SGRL")
        dataset.set_cl_embeds(subgraph_embeddings)
    
    optimizier = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.task == 'classification':
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        print(f"Task is {args.task}, using BCEWithLogitsLoss")

    elif args.loss == 'gai':
        gmm_path = train_gmm(dataset)
        criterion = GAILoss(init_noise_sigma=args.noise_sigma, gmm=gmm_path, device=device)
    elif args.loss == 'bmc':
        criterion = BMCLoss(init_noise_sigma=args.noise_sigma, device=device)
    elif args.loss == 'mse':
        criterion = torch.nn.MSELoss(reduction='mean')
    else:
        raise ValueError(f"Loss func {args.loss} not supported!")

    start = time.time()

    ## Start training, go go go!
    train(args, model, optimizier, criterion,
          train_loader, val_loader, test_loaders, 
          train_dspd_list, valid_dspd_list, 
          test_dspd_dict, device)
    
    elapsed = time.time() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    print(f"Done! Training took {timestr}")