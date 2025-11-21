import torch
from weighted_model import ConditionalUNet1D, ConditionalDDPM1D  # 修改为你的训练文件名
import argparse
import numpy as np
import json
import pandas as pd
import ast

def denormalize_data(normalized_data, mean=None, std=None, param_file="normalization_params.json"):
    """
    将归一化数据还原为原始数值范围
    参数：
        normalized_data: list[list[float]] 或 np.ndarray
        mean, std: 可手动传入；如果为 None，则从 param_file 读取
    返回：
        反归一化后的数据（list[list[float]]）
    """
    if mean is None or std is None:
        with open(param_file, "r") as f:
            params = json.load(f)
        mean, std = params["mean"], params["std"]

    # 保持结构
    denorm = [((np.array(seq, dtype=np.float32) * std) + mean).tolist() for seq in normalized_data]
    return denorm

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    torch.manual_seed(42)

    # ---------- 超参数（必须和训练时一致） ----------
    parser = argparse.ArgumentParser(description="DDPM Training Parameters")

    # 模型相关参数
    parser.add_argument("--station_emb_dim", type=int, default=64, help="Station embedding dimension")
    parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--base_channels", type=int, default=8, help="Base channels for UNet")
    parser.add_argument("--time_emb_dim", type=int, default=128, help="Time embedding dimension")
    parser.add_argument("--L", type=int, default=168, help="Sequence length")

    # 训练相关参数
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()

    # ---------- 创建模型实例 ----------
    unet = ConditionalUNet1D(
        station_emb_dim=args.station_emb_dim,
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        time_emb_dim=args.time_emb_dim
    ).to(device)
    ddpm = ConditionalDDPM1D(
        unet,
        seq_len=args.L,
        device=device,
        timesteps=args.timesteps
    ).to(device)

    # ---------- 加载训练好的参数 ----------
    ddpm.load_state_dict(torch.load("ddpm_model_nanjing.pth", map_location=device))
    ddpm.eval()  # 推理模式
    print("模型已加载成功！")

    # ---------- 测试生成 ----------
target_city = "yichang"

# 读取Excel文件
df = pd.read_excel('combined_data_with_embeddings_and_similarity.xlsx')

# 找到所有 file_name 为 "nanjing" 的行
nanjing_rows = df[df['file_name'] == f'{target_city}_hourly_demand.xlsx']

if not nanjing_rows.empty:
    # 处理所有embedding
    embeddings = []
    station_ids = []
    for _, row in nanjing_rows.iterrows():
        # 处理embedding
        emb = row['embedding']
        if isinstance(emb, str):
            embeddings.append(ast.literal_eval(emb))
        else:
            embeddings.append(emb)
        
        # 记录station_id
        station_ids.append(row['station_id'])
    
    station_emb_test = torch.tensor(embeddings, device=device, dtype=torch.float32)
    B = len(embeddings)  # B 等于实际找到的embedding数量
else:
    print(f"Warning: No row with file_name={target_city} found")



with torch.no_grad():
    generated = ddpm.sample(station_emb_test)
    print("Generated sequences shape:", generated.shape)

    # 创建结果DataFrame
    results = []
    
    for i in range(len(generated)):
        # 反归一化生成的数据
        denormalized = denormalize_data(generated[i].cpu())
        
        # 先打印数据类型和结构来调试
        print(f"\nSample {i+1} - Type: {type(denormalized)}")
        #print(f"Structure: {denormalized}")
        
        # 如果是嵌套列表，需要展平或按维度处理
        if isinstance(denormalized, list) and len(denormalized) > 0 and isinstance(denormalized[0], list):
            # 处理嵌套列表
            denormalized_non_negative = []
            for sublist in denormalized:
                if isinstance(sublist, list):
                    # 对子列表中的每个元素处理负值
                    processed_sublist = [max(0, x) if isinstance(x, (int, float)) else x for x in sublist]
                    denormalized_non_negative.append(processed_sublist)
                else:
                    denormalized_non_negative.append(max(0, sublist) if isinstance(sublist, (int, float)) else sublist)
        else:
            # 处理单层列表
            denormalized_non_negative = [max(0, x) if isinstance(x, (int, float)) else x for x in denormalized]

        
        # 将结果添加到列表中
        results.append({
            'station_id': station_ids[i],
            'weekly_data': denormalized_non_negative
        })
        
    
    # 创建DataFrame并保存到Excel
    results_df = pd.DataFrame(results)
    
    # 保存到Excel文件
    output_filename = f'./generated_data/{target_city}_generated_weekly_data.xlsx'
    results_df.to_excel(output_filename, index=False)
    print(f"\nResults saved to {output_filename}")
    print(f"Total {len(results)} stations' data saved")
