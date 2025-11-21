import torch
import torch.optim as optim
import copy
from weighted_model import ConditionalDDPM1D, ConditionalUNet1D
from tqdm import tqdm
import pandas as pd
import ast
import numpy as np
import json
import argparse

def batch_iter(x, station_emb, station_weight, batch_size, shuffle=True):
    B = x.shape[0]
    indices = torch.arange(B)
    if shuffle:
        indices = indices[torch.randperm(B)]
    for start in range(0, B, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        x_batch = x[batch_idx]
        station_emb_batch = station_emb[batch_idx]
        if station_weight is not None:
            station_weight_batch = station_weight[batch_idx]
        else:
            station_weight_batch = None
        yield x_batch, station_emb_batch, station_weight_batch


class DDPMTrainer:
    def __init__(self, unet, seq_len, station_emb_dim, timesteps, lr, device):
        self.device = device
        self.unet = unet.to(device)
        self.ddpm = ConditionalDDPM1D(self.unet, seq_len=seq_len, device=device, timesteps=timesteps)
        self.optimizer = optim.Adam(self.ddpm.parameters(), lr=lr)
        self.station_emb_dim = station_emb_dim

    

    def train(self, x_train, station_emb_train, station_weight_train, epochs, batch_size):
        x_train = x_train.to(self.device)
        station_emb_train = station_emb_train.to(self.device)
        if station_weight_train is not None:
            station_weight_train = station_weight_train.to(self.device)

        self.params_before = copy.deepcopy(self.ddpm.state_dict())

        for epoch in range(epochs):
            batch_iterable = batch_iter(x_train, station_emb_train, station_weight_train, batch_size)
            num_batches = (x_train.shape[0] + batch_size - 1) // batch_size
            epoch_loss = 0.0

            # tqdm 显示每个 epoch 的 batch 进度
            with tqdm(batch_iterable, total=num_batches, desc=f"Epoch {epoch+1}/{epochs}") as t:
                for x_batch, s_batch, w_batch in t:
                    self.optimizer.zero_grad()
                    loss = self.ddpm(x_batch, s_batch, w_batch)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    t.set_postfix({"batch_loss": loss.item()})

            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1} average loss: {avg_loss:.6f}")

        self.params_after = self.ddpm.state_dict()

    '''
    def compare_params(self):
        print("\nParameter changes (L2 norm):")
        for name in self.params_before:
            before = self.params_before[name]
            after = self.params_after[name]
            change = torch.norm(after - before).item()
            print(f"{name}: {change:.6f}")
    '''
    def test(self, station_emb, x_original=None, num_samples=2):
        station_emb = station_emb.to(self.device)
        with torch.no_grad():
            generated = self.ddpm.sample(station_emb)
            print("\nGenerated sequences shape:", generated.shape)

            if x_original is not None:
                for i in range(min(num_samples, x_original.shape[0])):
                    print(f"\nSample {i+1}:")
                    print("Original :", x_original[i].cpu().numpy())
                    print("Generated:", generated[i].cpu().numpy())
        return generated

def preprocess_data(weekly_data_list, embedding_list, weight_list):
    # 确保所有序列长度一致（取最小长度或填充）
    sequence_lengths = [len(seq) for seq in weekly_data_list]
    min_length = min(sequence_lengths)
    max_length = max(sequence_lengths)
    
    print(f"序列长度范围: {min_length} - {max_length}")
    
    # 如果长度不一致，可以选择截断或填充
    if min_length != max_length:
        print("警告: 序列长度不一致，进行截断处理")
        # 截断到最小长度
        weekly_data_processed = [seq[:min_length] for seq in weekly_data_list]
    else:
        weekly_data_processed = weekly_data_list
    
    # 转换为张量
    x_train = torch.tensor(weekly_data_processed, dtype=torch.float32).unsqueeze(1)  # (B, 1, L)
    station_emb_train = torch.tensor(embedding_list, dtype=torch.float32)            # (B, station_emb_dim)
    weight_train = torch.tensor(weight_list, dtype=torch.float32)
    return x_train, station_emb_train,weight_train

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


# ---------------- main ----------------
if __name__ == "__main__":
    #torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    parser = argparse.ArgumentParser(description="DDPM Training Parameters")

    # 模型相关参数
    parser.add_argument("--station_emb_dim", type=int, default=64, help="Station embedding dimension")
    parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--base_channels", type=int, default=8, help="Base channels for UNet")
    parser.add_argument("--time_emb_dim", type=int, default=128, help="Time embedding dimension")
    parser.add_argument("--L", type=int, default=168, help="Sequence length")

    # 训练相关参数
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for training")

    args = parser.parse_args()

    
    unet = ConditionalUNet1D(station_emb_dim=args.station_emb_dim, in_channels=args.in_channels, base_channels=args.base_channels, time_emb_dim=args.time_emb_dim)
    trainer = DDPMTrainer(unet, seq_len=args.L, station_emb_dim=args.station_emb_dim, timesteps=args.timesteps, lr=args.lr, device=device)

    #x_train = torch.randn(B, 1, L)
    df = pd.read_excel('combined_data_with_embeddings_and_similarity.xlsx')

    target_city = 'yichang'
# 提取 weekly_data 和 embedding 列
    target_index = df[df['file_name'] == f'nanjing_hourly_demand.xlsx'].index[0]

    # 只取目标索引之前的数据
    weekly_data_list = df.loc[:target_index-1, 'weekly_data'].tolist()
    embedding_list = df.loc[:target_index-1, 'embedding'].tolist()
    weight_list = df.loc[:target_index-1, f'sim_{target_city}'].tolist()


    weekly_data_list = [ast.literal_eval(seq) if isinstance(seq, str) else seq for seq in weekly_data_list]
    embedding_list = [ast.literal_eval(seq) if isinstance(seq, str) else seq for seq in embedding_list]
    B = len(weekly_data_list)
    print(f'-------------lenth of sample is {B}')

    # normalized
    all_data = np.concatenate([np.array(seq) for seq in weekly_data_list])
    mean = np.mean(all_data)
    std = np.std(all_data)
    std = std if std > 1e-8 else 1e-8

    normalized_weekly_data = [((np.array(seq) - mean) / std).tolist() for seq in weekly_data_list]
    norm_params = {'mean': float(mean), 'std': float(std)}
    with open("normalization_params.json", "w") as f:
        json.dump(norm_params, f)
    print("已保存归一化参数到 normalization_params.json")
    # weekly_data 列假设是 list-like 字符串，需要转换成列表
    # 如果是逗号分隔的字符串，可以这样处理：
    x_train, station_emb_train,station_weight_train = preprocess_data(normalized_weekly_data, embedding_list,weight_list)

    #station_emb_train = torch.randn(B, station_emb_dim)
    #station_weight_train = torch.rand(B)

    trainer.train(x_train, station_emb_train, station_weight_train, epochs=args.epochs, batch_size=args.batch_size)
    #trainer.compare_params()
    trainer.test(station_emb_train, x_original=x_train, num_samples=2)

    torch.save(trainer.ddpm.state_dict(), f"ddpm_model_{target_city}_ep1000.pth")
    print("DDPM 模型参数已保存到 ddpm_model_epoch1000.pth")
