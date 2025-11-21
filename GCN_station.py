import torch
import torch.nn as nn
import torch.nn.functional as F
from main_trainData_process import add_embedding_to_dict, StationMapper
import numpy as np
import pandas as pd
import json
import os

class MultiLayerGraphConv(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=2, activation=F.relu, use_batchnorm=True):
        """
        多层图卷积
        Args:
            in_features: 输入节点特征维度
            hidden_features: 隐藏层维度（每层相同）
            out_features: 输出节点特征维度
            num_layers: 图卷积层数（>=2）
            activation: 激活函数
            use_batchnorm: 是否使用 BatchNorm
        """
        super().__init__()
        assert num_layers >= 2, "num_layers至少为2"

        self.num_layers = num_layers
        self.activation = activation
        self.use_batchnorm = use_batchnorm

        # 构建多层图卷积
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None

        # 第一层
        self.convs.append(nn.Linear(in_features, hidden_features))
        if use_batchnorm:
            self.bns.append(nn.BatchNorm1d(hidden_features))

        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(nn.Linear(hidden_features, hidden_features))
            if use_batchnorm:
                self.bns.append(nn.BatchNorm1d(hidden_features))

        # 输出层
        self.convs.append(nn.Linear(hidden_features, out_features))
        if use_batchnorm:
            self.bns.append(nn.BatchNorm1d(out_features))

    def forward(self, x, adj):
        """
        x: [num_nodes, in_features] 节点特征
        adj: [num_nodes, num_nodes] 邻接矩阵或距离加权矩阵
        """
        # 对称归一化邻接矩阵
        deg_inv_sqrt = torch.pow(adj.sum(dim=1) + 1e-6, -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt  # [num_nodes, num_nodes]

        # 多层图卷积
        for i in range(self.num_layers):
            x = adj_norm @ x
            x = self.convs[i](x)
            if self.use_batchnorm:
                x = self.bns[i](x)
            if i != self.num_layers - 1:  # 最后一层可不加激活
                x = self.activation(x)
        return x

def process_city(station_embeddings_dict, distance_matrix_path, gcn_model, device='cuda'):
    """
    station_embeddings_dict: {station_id: embedding_tensor}
    distance_matrix_path: Excel 文件路径
    gcn_model: MultiLayerGraphConv 实例
    """
    # 1. 读取距离矩阵
    df = pd.read_excel(distance_matrix_path, index_col=0)
    station_ids = df.index.tolist()

    # 2. 构建节点特征 tensor [num_stations, embedding_dim]
    embedding_list = [station_embeddings_dict[sid].to(device) for sid in station_ids]
    x = torch.stack(embedding_list)

    # 3. 构建邻接矩阵
    distance_matrix = torch.tensor(df.values, dtype=torch.float32, device=device)
    adjacency_matrix = torch.exp(-distance_matrix)
    adjacency_matrix.fill_diagonal_(1.0)

    # 4. 图卷积
    out = gcn_model(x, adjacency_matrix)
    return {sid: out[i] for i, sid in enumerate(station_ids)}


# ----------------- 测试 -----------------
if __name__ == "__main__":
    num_stations = 5
    embedding_dim = 64
    hidden_dim = 64
    out_dim = 32
    num_layers = 3

    #dict_city_all = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cities = ["beijing", "chengdu", "dazhou", "deyang","Hohhot","nanchong","nanjing","shanghai","suining","yichang"]
    gcn_model = MultiLayerGraphConv(embedding_dim, hidden_dim, out_dim, num_layers).to(device)
# 遍历每个城市
    for city in cities:
        print(f"Processing city: {city} ...")
        
        # 1. 获取城市对应的映射
        mapper = StationMapper(f'./data_mapping/{city}_station_mapping.xlsx')
        mapping_dict = mapper.get_mapping()
        
        # 2 adding embedding
        embedding_array_raw = np.load(f'./pre_train/{city}_ukg_ER.npz')
        embedding_array = embedding_array_raw['E_pretrain']
        emb_dict = add_embedding_to_dict(mapping_dict, embedding_array)
        
        # 3 {stationID: embedding}
        new_dict = {v[0]: v[1] for v in emb_dict.values()} 
        #dict_city_all.update(new_dict)
        distance_file = f"./{city}_station_distance"
        city_out = process_city(new_dict, distance_file, gcn_model, device=device)

        # saving city-level embedding
        save_dir = "aggerated_embedding"
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{city}.json")
        with open(save_path, "w") as f:
            # 转为列表保存
            json.dump({sid: emb.cpu().tolist() for sid, emb in city_out.items()}, f)
        print(f"Saved aggregated embeddings to {save_path}")
        
        # 7. 计算城市平均 embedding
        embeddings_tensor = torch.stack(list(city_out.values()))
        avg_embedding = embeddings_tensor.mean(dim=0)
        
        # 8. 保存城市平均 embedding JSON
        avg_save_path = os.path.join(save_dir, f"{city}_avg.json")
        with open(avg_save_path, "w") as f:
            json.dump(avg_embedding.cpu().tolist(), f)
        print(f"Saved city-level average embedding to {avg_save_path}")

        # saving embeddings
        

