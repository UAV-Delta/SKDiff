import torch
import torch.nn.functional as F
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import ast
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
        print('this is weights ***************************')
        print(weights)

        return weights[0] if single_target else weights


def load_target_embedding_from_json(json_file_path, city):
    """从JSON文件加载目标嵌入"""
    with open(json_file_path, 'r') as f:
        target_data = json.load(f)
    target_embedding = target_data[city]
    return np.array(target_embedding, dtype=np.float32)


def extract_station_embeddings_by_city(df, city):
    """
    提取指定城市的 station_id 和 embedding 为字典
    """
    filtered_df = df[df['file_name'].str.contains(city, case=False, na=False)]
    
    result_dict = {}
    for _, row in filtered_df.iterrows():
        try:
            # 解析 embedding 字符串为列表
            embedding_list = ast.literal_eval(row['embedding'])
            result_dict[row['station_id']] = embedding_list
        except Exception as e:
            print(f"Error parsing embedding for station {row['station_id']}: {e}")
            result_dict[row['station_id']] = []  # 或者跳过这个站点
    
    return result_dict

def load_station_embeddings_from_dict(station_dict):
    """
    从字典中加载 station_id 和 embedding 为 numpy 数组
    """
    station_ids = list(station_dict.keys())
    embeddings = [station_dict[sid] for sid in station_ids]
    embeddings_np = np.array(embeddings, dtype=np.float32)
    
    return station_ids, embeddings_np



if __name__ == "__main__":
    # 示例 station embeddings dict
    all_results = [] 
    city_embedding_dict = {}
    cities = ["beijing", "chengdu", "dazhou", "deyang","Hohhot","nanchong","nanjing","shanghai","suining","yichang"]
    source_cities = ["beijing", "chengdu", "dazhou", "deyang","Hohhot","nanchong"]
    target_cities = ["nanjing","shanghai","suining","yichang"]
   # target_city = 'nanjing'
    for source_city in source_cities:
        for target_city in target_cities:
            print(f'Processing source_city: {source_city}...')

            json_file_path = f"./city_embeddings.json"  # 你的JSON文件路径
            
            '''
            mapper = StationMapper(f'./data_mapping/{source_city}_station_mapping.xlsx')
            mapping_dict = mapper.get_mapping()
            
            # 2 adding embedding
            embedding_array_raw = np.load(f'./pre_train/{source_city}_ukg_ER.npz')
            embedding_array = embedding_array_raw['E_pretrain']
            emb_dict = add_embedding_to_dict(mapping_dict, embedding_array)
            
            # 3 adding the new dict
            station_dict = {v[0]: v[1] for v in emb_dict.values()} 
            '''
            # 示例调用：
# 假设你的数据已经读入为 df
# beijing_stations = extract_station_embeddings_by_city(df, "beijing")
# print(beijing_stations)

            df = pd.read_excel("combined_data_with_embeddings.xlsx")
            station_dict = extract_station_embeddings_by_city(df, source_city)
            print("station_dic.keys() *****************")
            print(station_dict.keys())

            #print("*"*100)
            #print(station_dict.keys())

            # 从字典/JSON加载数据
            station_ids, station_embeddings_np = load_station_embeddings_from_dict(station_dict)
            target_embedding_np = load_target_embedding_from_json(json_file_path, target_city)

            # 转换为GPU张量
            station_embeddings = torch.tensor(station_embeddings_np, dtype=torch.float32, device='cuda')
            target_embeddings = torch.tensor(target_embedding_np, dtype=torch.float32, device='cuda')

            # 初始化计算器
            weight_calc = CosineSoftmaxWeightGPU(tau=0.2, device='cuda')

            # 计算权重
            weights = weight_calc.compute(station_embeddings, target_embeddings)
            weights_np = weights.cpu().detach().numpy()  # (num_stations,)
        
            df_city = pd.DataFrame({
                'source_city': source_city,
                'station_id': station_ids,
                'target_city': target_city,
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


# ---------- 配置（按需修改文件名/列名） ----------
combined_fp = "combined_data_with_embeddings.xlsx"
similarity_fp = "similarity_scores_all_cities.xlsx"
output_fp = "combined_data_with_embeddings_and_similarity.xlsx"

# 两表中 station_id 列名（如果两个表列名不同，请分别修改）
station_id_col_combined = "station_id"
station_id_col_similarity = "station_id"

# similarity 表中的列名
similarity_col = "similarity_score"
target_city_col = "target_city"

# 指定四个目标城市（严格按此顺序生成列）
expected_cities = ["nanjing", "shanghai", "suining", "yichang"]

# 聚合方法（防止重复记录出现）
agg_func = "mean"  # 若重复，可改为 'max' 等

# 生成相似度列的前缀
prefix = "sim_"
# -------------------------------------------------

# 读取文件
df_combined = pd.read_excel(combined_fp)
df_sim = pd.read_excel(similarity_fp)

# 校验关键列
for col, dfname in [
    (station_id_col_combined, "combined"),
    (station_id_col_similarity, "similarity"),
    (similarity_col, "similarity"),
    (target_city_col, "similarity"),
]:
    df = df_combined if dfname == "combined" else df_sim
    if col not in df.columns:
        raise KeyError(f"在 `{dfname}` 表中找不到列 `{col}`，请检查列名并修改脚本配置。")

# 如果两个表的 station_id 列名不同，先重命名 similarity 表的列到 combined 一致
if station_id_col_similarity != station_id_col_combined:
    df_sim = df_sim.rename(columns={station_id_col_similarity: station_id_col_combined})
station_id_col = station_id_col_combined

# 为防止类型不一致（例如一个是数值一个是字符串），统一把 station_id 转为字符串
df_combined[station_id_col] = df_combined[station_id_col].astype(str)
df_sim[station_id_col] = df_sim[station_id_col].astype(str)

# 聚合 (station_id, target_city) -> 单一相似度（稳健处理）
df_sim_agg = (
    df_sim
    .groupby([station_id_col, target_city_col], as_index=False)[similarity_col]
    .agg(agg_func)
)

# 检查 similarity 表里是否包含你指定的四个城市
actual_cities = sorted(df_sim_agg[target_city_col].unique().tolist())
missing = [c for c in expected_cities if c not in actual_cities]
if missing:
    print(f"警告：在 similarity 文件中未找到以下期望的目标城市：{missing}")
    # 仍然继续，但对应缺失城市列会全部为 NaN

extra = [c for c in actual_cities if c not in expected_cities]
if extra:
    print(f"提示：similarity 文件中还有这些未在 expected_cities 列表内的城市（会被忽略）：{extra}")

# 只保留 expected_cities 中的记录（如果某些期望城市不存在，则相应列会缺失）
df_sim_agg = df_sim_agg[df_sim_agg[target_city_col].isin(expected_cities)]

# pivot 展开为列
df_pivot = df_sim_agg.pivot(index=station_id_col, columns=target_city_col, values=similarity_col)

# 确保列顺序为 expected_cities 顺序，并加前缀
ordered_cols = [c for c in expected_cities if c in df_pivot.columns]
# 若某些期望城市完全没有在数据中出现，则它们不会在 df_pivot.columns 中；我们仍要创建这些列（值为 NaN）
for c in expected_cities:
    col_name = c
    if col_name not in df_pivot.columns:
        df_pivot[col_name] = pd.NA

# 现在按 expected_cities 顺序重排
df_pivot = df_pivot[expected_cities]
df_pivot.columns = [f"{prefix}{c}" for c in df_pivot.columns]

# 把 index 还原为列以便合并
df_pivot = df_pivot.reset_index()

# 合并（以 combined 为主表，left join）
df_merged = df_combined.merge(df_pivot, on=station_id_col, how="left")

# 保存结果
df_merged.to_excel(output_fp, index=False)
print(f"已保存合并结果为：{output_fp}")
print(f"combined 原记录数：{len(df_combined)}，合并后记录数：{len(df_merged)}")

# 显示新增列并给出统计信息
sim_columns = [f"{prefix}{c}" for c in expected_cities]
print("生成的相似度列（按顺序）：", sim_columns)
for col in sim_columns:
    if col in df_merged.columns:
        print(f"\n列 {col} 的统计：")
        print(df_merged[col].describe())
    else:
        print(f"\n列 {col} 未出现在合并后的表中（该目标城市在相似度文件中完全缺失）。")

# 可以选择将 NaN 填为 0（若需要，请取消下面一行注释）
# df_merged[sim_columns] = df_merged[sim_columns].fillna(0)

