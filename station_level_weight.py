import torch
import torch.nn.functional as F
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
from main_trainData_process import add_embedding_to_dict, StationMapper

class CosineSoftmaxWeightGPU:
    """
    GPU 版余弦相似度 + softmax 权重计算器
    支持批量目标、多候选 station
    """

    def __init__(self, tau: float = 0.2, eps: float = 1e-12, device='cuda'):
        self.tau = tau
        self.eps = eps
        self.device = device if torch.cuda.is_available() else 'cpu'

    def _l2_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps))

    def compute(self, station_embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
        station_embeddings = station_embeddings.to(self.device)
        target_embeddings = target_embeddings.to(self.device)

        single_target = (target_embeddings.dim() == 1)
        if single_target:
            target_embeddings = target_embeddings.unsqueeze(0)  # (1, embedding_dim)

        station_embeddings = self._l2_normalize(station_embeddings)
        target_embeddings = self._l2_normalize(target_embeddings)

        # 计算余弦相似度
        scores = target_embeddings.matmul(station_embeddings.t()) / (self.tau + self.eps)  # (num_targets, num_stations)

        # softmax 归一化
        weights = F.softmax(scores, dim=1)

        return weights[0] if single_target else weights

def load_station_embeddings_from_dict(station_dict):
    """从字典加载站点嵌入"""
    station_ids = list(station_dict.keys())
    embeddings = np.array([station_dict[sid] for sid in station_ids], dtype=np.float32)
    return station_ids, embeddings

def load_target_embedding_from_json(json_file_path, city):
    """从JSON文件加载目标嵌入"""
    with open(json_file_path, 'r') as f:
        target_data = json.load(f)
    target_embedding = target_data[city]
    return np.array(target_embedding, dtype=np.float32)



if __name__ == "__main__":
    # 示例 station embeddings dict
    all_results = [] 

    cities = ["beijing", "chengdu", "dazhou", "deyang","Hohhot","nanchong","nanjing","shanghai","suining","yichang"]
    for city in cities:
        print(f'Processing city: {city}...')

        json_file_path = f"./aggregating_embeddings/{city}.json"  # 你的JSON文件路径

        mapper = StationMapper(f'./data_mapping/{city}_station_mapping.xlsx')
        mapping_dict = mapper.get_mapping()
        
        # 2 adding embedding
        embedding_array_raw = np.load(f'./pre_train/{city}_ukg_ER.npz')
        embedding_array = embedding_array_raw['E_pretrain']
        emb_dict = add_embedding_to_dict(mapping_dict, embedding_array)
        
        # 3 adding the new dict
        station_dict = {v[0]: v[1] for v in emb_dict.values()} 
        

        # 从字典/JSON加载数据
        station_ids, station_embeddings_np = load_station_embeddings_from_dict(station_dict)
        target_embedding_np = load_target_embedding_from_json(json_file_path, city)

        # 转换为GPU张量
        station_embeddings = torch.tensor(station_embeddings_np, dtype=torch.float32, device='cuda')
        target_embeddings = torch.tensor(target_embedding_np, dtype=torch.float32, device='cuda')

        # 初始化计算器
        weight_calc = CosineSoftmaxWeightGPU(tau=0.2, device='cuda')

        # 计算权重
        weights = weight_calc.compute(station_embeddings, target_embeddings)
        weights_np = weights.cpu().detach().numpy()  # (num_stations,)

        df_city = pd.DataFrame({
            'city': city,
            'station_id': station_ids,
            'similarity_score': weights_np
        })
        all_results.append(df_city)
        df_all = pd.concat(all_results, ignore_index=True)

    # 写入 Excel
    output_file_path = "similarity_scores_all_cities.xlsx"
    df_all.to_excel(output_file_path, index=False)
    print(f"所有城市相似度分数已保存到 {output_file_path}")


    #################################################


    # 读取两个 Excel 文件
    df_combined = pd.read_excel("combined_data_with_embeddings.xlsx")
    df_similarity = pd.read_excel("similarity_scores_all_cities.xlsx")

    # 查看相似度文件的列名，确认 station_id 和相似度分数的列名
    print("相似度文件列名:", df_similarity.columns.tolist())

    # 假设相似度文件中 station_id 列名为 'station_id'，相似度列名为 'similarity_score'
    # 如果不是，请根据实际情况修改下面的列名
    station_id_col = 'station_id'  # 请根据实际情况修改
    similarity_col = 'similarity_score'  # 请根据实际情况修改

    # 合并数据，将相似度分数添加到 combined_data 中
    df_merged = df_combined.merge(
        df_similarity[[station_id_col, similarity_col]], 
        on=station_id_col, 
        how='left'
    )

    # 保存结果
    df_merged.to_excel("combined_data_with_embeddings_and_similarity.xlsx", index=False)

    print(f"成功匹配并保存！共处理 {len(df_merged)} 条记录")
    print(f"相似度分数统计:")
    print(df_merged[similarity_col].describe())
