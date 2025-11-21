import pandas as pd
import numpy as np
import torch
from typing import Dict, List
import os
from itertools import chain
import json


class StationMapper:
    def __init__(self, file_path):
        self.mapping = self._load_mapping(file_path)
    
    def _load_mapping(self, file_path):
        """加载映射关系，key为右侧编码，value为左侧原始code"""
        df = pd.read_excel(file_path)
        mapping_dict = {}
        
        for _, row in df.iterrows():
            original_code = str(row['原始code']).strip()
            new_code = str(row['编码']).strip()
            
            # 跳过表头
            if original_code != '原始code' and new_code != '编码':
                mapping_dict[new_code] = original_code
        
        return mapping_dict
    
    def get_mapping(self):
        """返回完整的映射字典"""
        return self.mapping


def add_embedding_to_dict(mapping_dict, embedding_array):
    """
    将embedding添加到字典中，每个key对应一个元组(code, embedding)
    """
    new_dict = {}
    
    stations = list(mapping_dict.keys())
    for i, station in enumerate(stations):
        new_dict[station] = (mapping_dict[station], embedding_array[i])
    
    return new_dict



def extract_and_filter_weekly_data(file_path: str, sum_threshold: int = 100, zero_ratio_threshold: float = 0.6):
    """
    从 Excel 文件中提取数据，并过滤掉不符合条件的元素
    
    参数:
        file_path: Excel 文件路径
        sum_threshold: 总和阈值，小于等于此值的会被剔除
        zero_ratio_threshold: 零值比例阈值，大于等于此值的会被剔除
    
    返回:
        过滤后的字典列表
    """
    df = pd.read_excel(file_path)
    
    timestamp_col = df.columns[0]
    station_cols = df.columns[1:]
    
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df['week'] = df[timestamp_col].dt.to_period('W')
    
    weekly_dicts = []
    
    for week, week_df in df.groupby('week'):
        # 确保是完整的一周（168小时）
        if len(week_df) == 168:
            # 为这一周的每个站点创建一个字典
            for station in station_cols:
                station_data = week_df[station].tolist()
                
                # 计算数据特征
                data_sum = sum(station_data)
                zero_count = station_data.count(0)
                zero_ratio = zero_count / len(station_data)
                
                # 过滤条件：总和 > threshold 且 零值比例 < threshold
                if data_sum > sum_threshold and zero_ratio < zero_ratio_threshold:
                    weekly_dicts.append({str(station): station_data})
    
    return weekly_dicts


def process_and_combine_to_dataframe(file_paths: List[str], sum_threshold: int = 100, zero_ratio_threshold: float = 0.6) -> pd.DataFrame:
    """
    处理多个Excel文件并将结果转换为结构化的DataFrame
    
    参数:
        file_paths: Excel文件路径列表
        sum_threshold: 总和阈值
        zero_ratio_threshold: 零值比例阈值
    
    返回:
        包含所有数据的DataFrame
    """
    all_data = []
    
    for file_path in file_paths:
        print(f"正在处理文件: {file_path}")
        try:
            result = extract_and_filter_weekly_data(file_path, sum_threshold, zero_ratio_threshold)
            
            # 转换数据结构
            for item in result:
                for station, data in item.items():
                    all_data.append({
                        'file_name': os.path.basename(file_path),
                        'station': station,
                        'weekly_data': data,
                        'data_length': len(data),
                        'data_sum': sum(data),
                        'zero_count': data.count(0),
                        'zero_ratio': data.count(0) / len(data)
                    })
            
            print(f"  成功处理，得到 {len(result)} 条记录")
        except Exception as e:
            print(f"  处理文件 {file_path} 时出错: {e}")
    
    return pd.DataFrame(all_data)





def process_and_combine_with_embeddings(file_paths: List[str], 
                                      embeddings_dict: Dict,
                                      sum_threshold: int = 100, 
                                      zero_ratio_threshold: float = 0.5) -> pd.DataFrame:
    """
    处理多个Excel文件并将结果转换为包含embedding的DataFrame
    
    参数:
        file_paths: Excel文件路径列表
        embeddings_dict: station embedding字典 (key: station_id, value: embedding tensor)
        sum_threshold: 总和阈值
        zero_ratio_threshold: 零值比例阈值
    
    返回:
        包含所有数据和embedding的DataFrame
    """
    
    all_data = []
    
    for file_path in file_paths:
        print(f"正在处理文件: {file_path}")
        try:
            # 使用你原有的函数
            result = extract_and_filter_weekly_data(file_path, sum_threshold, zero_ratio_threshold)
            
            # 转换数据结构并添加embedding
            for item in result:
                for station_id, data in item.items():
                    # 直接通过station_id获取embedding
                    embedding_tensor = embeddings_dict.get(station_id)
                    
                    if embedding_tensor is not None:
                        # 将tensor转换为列表
                        embedding_list = embedding_tensor.tolist()
                        all_data.append({
                            'file_name': os.path.basename(file_path),
                            'station_id': station_id,
                            'weekly_data': data,
                            'data_length': len(data),
                            'data_sum': sum(data),
                            'zero_count': data.count(0),
                            'zero_ratio': data.count(0) / len(data),
                            'embedding': embedding_list
                        })
                    else:
                        print(f"  警告: station {station_id} 在embedding数据中未找到")
                        # 如果没有embedding，仍然保存其他数据
                        all_data.append({
                            'file_name': os.path.basename(file_path),
                            'station_id': station_id,
                            'weekly_data': data,
                            'data_length': len(data),
                            'data_sum': sum(data),
                            'zero_count': data.count(0),
                            'zero_ratio': data.count(0) / len(data),
                            'embedding': None
                        })
            
            print(f"  成功处理，得到 {len(result)} 条记录")
        except Exception as e:
            print(f"  处理文件 {file_path} 时出错: {e}")
    
    return pd.DataFrame(all_data)

    



# 使用示例

if __name__ == "__main__":
     
    
    cities = ["beijing", "chengdu", "dazhou", "deyang","Hohhot","nanchong","nanjing","shanghai","suining","yichang"]
    demand_file_paths = [f"./dataset/{city}_hourly_demand.xlsx" for city in cities]
    # get dict
    '''
    mapper = StationMapper(f'./data_mapping/{city}_station_mapping.xlsx')
    mapping_dict = mapper.get_mapping()
    dictfileter = DictFilter(mapper)
    embedding_dict = dictfileter.filter_by_keyword()

    embedding_array = np.load(f'./pre_train/{city}_ukg_ER.npz')
    enhanced_dict = add_embedding_to_dict(mapping_dict, embedding_array)
    '''
    dict_city_all = {}
    city_embedding_dict = {}
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
        
        # 3 adding the new dict
        new_dict = {v[0]: v[1] for v in emb_dict.values()} 
        dict_city_all.update(new_dict)

        station_embeddings = np.array(list(new_dict.values()))

        # 求城市平均 embedding
        city_embedding = np.mean(station_embeddings, axis=0)

        # 转成 Python list，方便序列化为 JSON
        city_embedding_dict[city] = city_embedding.tolist()

    output_path = "./city_embeddings.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(city_embedding_dict, f, ensure_ascii=False, indent=4)

    print(f"City embeddings saved to {output_path}")

    print("开始处理文件并添加embedding...")
    combined_df = process_and_combine_with_embeddings(demand_file_paths, dict_city_all)
    
    # 查看结果
    print("\n处理完成！")
    print(f"总记录数: {len(combined_df)}")
    
    # 检查embedding匹配情况
    stations_with_embedding = combined_df[combined_df['embedding'].notnull()]
    stations_without_embedding = combined_df[combined_df['embedding'].isnull()]
    
    print(f"\n有embedding的station数量: {len(stations_with_embedding)}")
    print(f"无embedding的station数量: {len(stations_without_embedding)}")
    
    if len(stations_without_embedding) > 0:
        missing_stations = stations_without_embedding['station_id'].unique()
        print(f"未匹配到embedding的station ID ({len(missing_stations)}个):")
        print(missing_stations)
    
    # 显示前几条记录作为示例
    print("\n前3条记录示例:")
    for i, row in combined_df.head(3).iterrows():
        print(f"Station: {row['station_id']}, 数据长度: {row['data_length']}, Embedding: {row['embedding']}")
    
    # 保存到Excel
    output_file = "combined_data_with_embeddings.xlsx"
    combined_df.to_excel(output_file, index=False)
    print(f"\n结果已保存到 {output_file}")

'''
# 使用示例
if __name__ == "__main__":
   # 使用示例
    mapper = StationMapper('./data_mapping/beijing_station_mapping.xlsx')
    mapping_dict = mapper.get_mapping()
    #print(mapping_dict)

    # 使用示例
    embedding_array = torch.randn([25, 2])  # 你的embedding数组
    enhanced_dict = add_embedding_to_dict(mapping_dict, embedding_array)

    # 查看结果
    print(enhanced_dict)
    # 输出: {'original_code': '582', 'embedding': array([0.123, 0.456, ...])}
  ''' 


'''
class DictFilter:
    def __init__(self, data):
        self.data = data

    def filter_by_keyword(self, keyword = 'station'):
        """
        返回过滤后的字典，只保留键中包含 keyword 的项
        """
        return {k: v for k, v in self.data.items() if keyword in k}
'''