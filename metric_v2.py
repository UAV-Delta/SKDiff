# compute_metrics_nanjing.py
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
import argparse

# --------- Strict parsing for 168 values ----------
def parse_weekly_data(cell) -> np.ndarray:
    """
    严格解析周数据，必须是168个数字
    处理两层列表格式 [[]]
    """
    if pd.isna(cell):
        return None
    
    try:
        # 处理两层列表的情况
        if isinstance(cell, (list, tuple, np.ndarray)):
            if len(cell) > 0 and isinstance(cell[0], (list, tuple, np.ndarray)):
                # 取内层列表
                arr = np.array(cell[0], dtype=float)
            else:
                arr = np.array(cell, dtype=float)
            
            # 严格检查长度
            if len(arr) != 168:
                print(f"Warning: Expected 168 values, got {len(arr)}")
                return None
            return arr
        
        # 处理字符串格式
        if isinstance(cell, str):
            cell = cell.strip()
            # 解析为列表
            import ast
            parsed = ast.literal_eval(cell)
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
                arr = np.array(parsed[0], dtype=float)
            else:
                arr = np.array(parsed, dtype=float)
            
            # 严格检查长度
            if len(arr) != 168:
                print(f"Warning: Expected 168 values, got {len(arr)}")
                return None
            return arr
                
    except Exception as e:
        print(f"Parse error: {e}")
        return None
    
    return None

# --------- Metrics implementations ----------
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 1e-8
    if not np.any(mask):
        return float('nan')
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100.0

def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon距离"""
    if p.shape != q.shape:
        raise ValueError("p and q must have same shape")
    
    p = np.maximum(p, 0) + 1e-12
    q = np.maximum(q, 0) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    
    return float(jensenshannon(p, q, base=2.0))

def mmd_rbf_simple(x: np.ndarray, y: np.ndarray) -> float:
    """MMD计算"""
    x = x.flatten()
    y = y.flatten()
    
    # 中位数启发式带宽
    z = np.concatenate([x, y])
    if z.size <= 1:
        return 1.0
    sigma = np.median(np.abs(z[:, None] - z[None, :]))
    if sigma <= 0:
        sigma = 1.0
    
    def gaussian_kernel(a, b, sigma):
        dists_sq = np.sum((a[:, None] - b[None, :]) ** 2, axis=2)
        return np.exp(-dists_sq / (2 * sigma ** 2))
    
    Kxx = gaussian_kernel(x.reshape(-1, 1), x.reshape(-1, 1), sigma)
    Kyy = gaussian_kernel(y.reshape(-1, 1), y.reshape(-1, 1), sigma)
    Kxy = gaussian_kernel(x.reshape(-1, 1), y.reshape(-1, 1), sigma)
    
    n, m = len(x), len(y)
    mmd_sq = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) + \
             (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) - \
             2 * Kxy.mean()
    
    return float(np.sqrt(max(mmd_sq, 0)))

# --------- Main processing ----------
def compute_metrics_for_nanjing(true_xlsx: str, pred_xlsx: str, output_xlsx: str, target_city: str):
    # 读取文件
    df_true = pd.read_excel(true_xlsx)
    df_pred = pd.read_excel(pred_xlsx)
    
    # 过滤目标城市数据
    true_file_marker = f'{target_city}_hourly_demand.xlsx'
    df_true_nj = df_true[df_true['file_name'] == true_file_marker].copy()
    
    if 'file_name' in df_pred.columns:
        df_pred_nj = df_pred[df_pred['file_name'] == true_file_marker].copy()
    else:
        df_pred_nj = df_pred.copy()
    
    print(f"Found {len(df_true_nj)} true samples and {len(df_pred_nj)} pred samples")
    
    # 解析数据并严格检查长度
    df_true_nj['true_array'] = df_true_nj['weekly_data'].apply(parse_weekly_data)
    df_pred_nj['pred_array'] = df_pred_nj['weekly_data'].apply(parse_weekly_data)
    
    # 统计长度信息
    true_lengths = df_true_nj['true_array'].apply(lambda x: len(x) if x is not None else 0)
    pred_lengths = df_pred_nj['pred_array'].apply(lambda x: len(x) if x is not None else 0)
    
    print(f"True data lengths: {true_lengths.value_counts().to_dict()}")
    print(f"Pred data lengths: {pred_lengths.value_counts().to_dict()}")
    
    # 移除无法解析或长度不为168的行
    df_true_nj = df_true_nj[df_true_nj['true_array'].apply(lambda x: x is not None and len(x) == 168)]
    df_pred_nj = df_pred_nj[df_pred_nj['pred_array'].apply(lambda x: x is not None and len(x) == 168)]
    
    print(f"After filtering: {len(df_true_nj)} true samples and {len(df_pred_nj)} pred samples")
    
    # 确保数据长度一致
    min_len = min(len(df_true_nj), len(df_pred_nj))
    df_true_nj = df_true_nj.head(min_len)
    df_pred_nj = df_pred_nj.head(min_len)
    
    # 计算指标
    metrics = ['MAE', 'MSE', 'MAPE', 'MMD', 'JSD']
    for metric in metrics:
        df_pred_nj[metric] = np.nan
    
    valid_count = 0
    for idx in range(min_len):
        y_true = df_true_nj.iloc[idx]['true_array']
        y_pred = df_pred_nj.iloc[idx]['pred_array']
        
        # 双重检查长度
        if len(y_true) != 168 or len(y_pred) != 168:
            print(f"Row {idx}: length error (true: {len(y_true)}, pred: {len(y_pred)})")
            continue
        
        try:
            df_pred_nj.at[df_pred_nj.index[idx], 'MAE'] = mae(y_true, y_pred)
            df_pred_nj.at[df_pred_nj.index[idx], 'MSE'] = mse(y_true, y_pred)
            df_pred_nj.at[df_pred_nj.index[idx], 'MAPE'] = mape(y_true, y_pred)
            df_pred_nj.at[df_pred_nj.index[idx], 'MMD'] = mmd_rbf_simple(y_true, y_pred)
            df_pred_nj.at[df_pred_nj.index[idx], 'JSD'] = js_distance(y_true, y_pred)
            valid_count += 1
        except Exception as e:
            print(f"Row {idx}: error computing metrics: {e}")
    
    # 保存结果
    df_pred_nj['true_weekly_data'] = df_true_nj['true_array'].values
    df_pred_nj.to_excel(output_xlsx, index=False)
    print(f"Saved metrics for {valid_count} valid samples to {output_xlsx}")
    
    if valid_count > 0:
        print(f"Metrics summary (based on {valid_count} samples):")
        for metric in metrics:
            if metric in df_pred_nj.columns:
                values = df_pred_nj[metric].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    print(f"  {metric}: {mean_val:.4f}")

if __name__ == "__main__":
    
    target_city="yichang"
    true_xlsx="combined_data_with_embeddings_and_similarity.xlsx"
    pred_xlsx=f"./generated_data/{target_city}_generated_weekly_data.xlsx"
    output_xlsx=f"./metric/{target_city}_metric.xlsx"
    


    compute_metrics_for_nanjing(true_xlsx, pred_xlsx, output_xlsx, target_city)