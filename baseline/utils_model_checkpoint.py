import os
import torch
import json
import datetime

def check_model_exists(model_name, dataset_name, params=None):
    """
    检查指定参数的模型是否已经存在
    
    参数:
        model_name (str): 模型名称 (如 'GraphCL', 'Cirgps', 'DLPL-CAP', 'paragraph-simple')
        dataset_name (str): 数据集名称
        params (dict): 模型参数字典，用于标识模型
    
    返回:
        tuple: (模型存在标志, 模型路径), 如果模型不存在，模型路径为应保存的路径
    """
    # 确保模型保存目录存在
    checkpoint_dir = os.path.join(f"./models_{model_name.lower()}", dataset_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 如果没有参数，尝试加载最新的模型
    if params is None:
        model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
        if not model_files:
            return False, os.path.join(checkpoint_dir, f"{model_name.lower()}_latest.pth")
        
        # 按照修改时间排序，找出最新的模型
        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
        return True, os.path.join(checkpoint_dir, model_files[0])
    
    if model_name == 'GraphCL':
    # 只关注指定的几个关键参数
        important_params = {
        'num_hops': 'nh',
        'task': 'tsk',
        'num_gnn_layers': 'ngl',
        'num_head_layers': 'nhl',
        'hid_dim': 'hd',
        'use_stats': 'us',
        'use_bn': 'bn',
        'dropout': 'dp',
        'aug1': 'aug1',
        'aug2': 'aug2',
        'aug_ratio1': 'ar1',
        'aug_ratio2': 'ar2',
        'gnn_type': 'gt',       
        }
    else:
        important_params = {
        'num_hops': 'nh',
        'task': 'tsk',
        'num_gnn_layers': 'ngl',
        'num_head_layers': 'nhl',
        'hid_dim': 'hd',
        'use_stats': 'us',
        'use_bn': 'bn',
        'dropout': 'dp',
        }
    
    param_str = ""
    for full_name, short_name in important_params.items():
        if full_name in params:
            value = params[full_name]
            param_str += f"{short_name}_{value}_"
    
    param_str = param_str.rstrip("_")
    
    # 构建模型文件名模式
    model_pattern = f"{model_name.lower()}_{param_str}"
    
    # 查找匹配的模型文件
    model_files = [f for f in os.listdir(checkpoint_dir) 
                  if f.startswith(model_pattern) and f.endswith(".pth")]
    
    if not model_files:
        return False, os.path.join(checkpoint_dir, f"{model_pattern}_latest.pth")
    
    # 按照修改时间排序，找出最新的模型
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    latest_model = model_files[0]
    
    # 优先使用best模型
    best_model = f"{model_pattern}_best.pth"
    if best_model in model_files:
        return True, os.path.join(checkpoint_dir, best_model)
    
    return True, os.path.join(checkpoint_dir, latest_model)

def save_model(model, model_name, dataset_name, epoch, params=None, is_best=False, metrics=None):
    """
    保存模型权重和训练信息
    
    参数:
        model: 要保存的模型
        model_name (str): 模型名称 (如 'GraphCL', 'Cirgps', 'DLPL-CAP', 'paragraph-simple')
        dataset_name (str): 数据集名称
        epoch (int): 当前训练轮数
        params (dict): 模型参数字典，用于标识模型
        is_best (bool): 是否是目前最优模型
        metrics (dict): 训练指标，如损失和准确率
    """
    checkpoint_dir = os.path.join(f"./models_{model_name.lower()}", dataset_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 保存带有时间戳和参数的模型
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 构建基础文件名
    if params:
        # 只关注指定的几个关键参数
        important_params = {
            'num_hops': 'nh',
            'task': 'tsk',
            'num_gnn_layers': 'ngl',
            'num_head_layers': 'nhl',
            'hid_dim': 'hd',
            'use_stats': 'us',
            'use_bn': 'bn',
            'dropout': 'dp'
        }
        
        param_str = ""
        for full_name, short_name in important_params.items():
            if full_name in params:
                value = params[full_name]
                param_str += f"{short_name}_{value}_"
        
        param_str = param_str.rstrip("_")
        
        base_filename = f"{model_name.lower()}_{param_str}"
    else:
        base_filename = f"{model_name.lower()}"
    
    # 确保文件名总长度不超过限制（通常为255个字符）
    max_filename_length = 200  # 留一些余量给目录路径
    if len(f"{base_filename}_epoch_{epoch}_{timestamp}.pth") > max_filename_length:
        # 截断基础文件名
        available_length = max_filename_length - len(f"_epoch_{epoch}_{timestamp}.pth")
        base_filename = base_filename[:available_length]
    
    # 保存当前模型
    model_path = os.path.join(checkpoint_dir, f"{base_filename}_epoch_{epoch}_{timestamp}.pth")
    torch.save(model.state_dict(), model_path)
    
    # 同时保存为最新模型
    latest_path = os.path.join(checkpoint_dir, f"{base_filename}_latest.pth")
    torch.save(model.state_dict(), latest_path)
    
    # 如果是最佳模型，另存一份
    if is_best:
        best_path = os.path.join(checkpoint_dir, f"{base_filename}_best.pth")
        torch.save(model.state_dict(), best_path)
    
    # 保存训练信息
    if metrics:
        info = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "epoch": epoch,
            "timestamp": timestamp,
            "params": params,
            "metrics": metrics,
            "is_best": is_best
        }
        info_path = os.path.join(checkpoint_dir, f"{base_filename}_info_{timestamp}.json")
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
    
    return model_path

def load_model(model, model_name, dataset_name, params=None, load_best=False):
    """
    加载已保存的模型权重
    
    参数:
        model: 要加载权重的模型
        model_name (str): 模型名称 (如 'GraphCL', 'Cirgps', 'DLPL-CAP', 'paragraph-simple')
        dataset_name (str): 数据集名称
        params (dict): 模型参数字典，用于标识模型
        load_best (bool): 是否加载最佳模型，False则加载最新模型
    
    返回:
        bool: 是否成功加载模型
    """
    checkpoint_dir = os.path.join(f"./models_{model_name.lower()}", dataset_name)
    
    if not os.path.exists(checkpoint_dir):
        print(f"模型目录不存在: {checkpoint_dir}")
        return False
    
    if params:
        # 只关注指定的几个关键参数
        important_params = {
            'num_hops': 'nh',
            'task': 'tsk',
            'num_gnn_layers': 'ngl',
            'num_head_layers': 'nhl',
            'hid_dim': 'hd',
            'use_stats': 'us',
            'use_bn': 'bn',
            'dropout': 'dp'
        }
        
        param_str = ""
        for full_name, short_name in important_params.items():
            if full_name in params:
                value = params[full_name]
                param_str += f"{short_name}_{value}_"
        
        param_str = param_str.rstrip("_")
        
        base_filename = f"{model_name.lower()}_{param_str}"
    else:
        base_filename = f"{model_name.lower()}"
    
    # 确定要加载的模型路径
    if load_best:
        model_path = os.path.join(checkpoint_dir, f"{base_filename}_best.pth")
        if not os.path.exists(model_path):
            print(f"最佳模型不存在: {model_path}")
            return False
    else:
        model_path = os.path.join(checkpoint_dir, f"{base_filename}_latest.pth")
        if not os.path.exists(model_path):
            # 尝试寻找最新的模型文件
            model_files = [f for f in os.listdir(checkpoint_dir) 
                          if f.startswith(base_filename) and f.endswith(".pth")]
            if not model_files:
                print(f"未找到匹配的模型文件: {base_filename}*.pth")
                return False
            
            # 按照修改时间排序，找出最新的模型
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
            model_path = os.path.join(checkpoint_dir, model_files[0])
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        print(f"成功加载模型: {model_path}")
        return True
    except Exception as e:
        print(f"加载模型失败: {e}")
        return False

def get_model_params_dict(args, exclude_keys=None):
    """
    从args中提取模型相关参数，创建一个字典用于标识模型
    
    参数:
        args: 参数对象
        exclude_keys (list): 要排除的参数名列表
    
    返回:
        dict: 模型参数字典
    """
    if exclude_keys is None:
        exclude_keys = ['num_workers', 'gpu', 'epochs', 'batch_size', 'seed']
    
    params = {}
    for key, value in vars(args).items():
        if key not in exclude_keys and not key.startswith('_'):
            params[key] = value
    
    return params 