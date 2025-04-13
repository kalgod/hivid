import os
import json  # 用于保存和加载字典
import numpy as np

def load_npy_files(directory):
    """
    遍历指定目录，读取每个子文件夹中的 llmscore_modified.npy 文件，并存储到一个字典中。
    :param directory: 根目录路径
    :return: 包含文件夹名和对应 llmscore_modified.npy 数据的字典
    """
    npy_dict = {}
    
    # 遍历指定目录中的每个文件夹
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        
        # 确保是目录
        if os.path.isdir(folder_path):
            npy_file_path = os.path.join(folder_path, 'llmscore.npy')
            
            # 检查 llmscore_modified.npy 文件是否存在
            if os.path.isfile(npy_file_path):
                try:
                    # 读取 npy 文件
                    npy_data = np.load(npy_file_path, allow_pickle=True)
                    npy_dict[folder_name] = npy_data.tolist()  # 转换为列表以便 JSON 序列化
                except Exception as e:
                    print(f"Error loading {npy_file_path}: {e}")
    
    return npy_dict

def save_dict_to_file(data_dict, file_path):
    """
    将字典保存到 JSON 文件中。
    :param data_dict: 要保存的字典
    :param file_path: 保存文件的路径
    """
    with open(file_path, 'w') as json_file:
        json.dump(data_dict, json_file)
    print(f"Dictionary saved to {file_path}")

def load_dict_from_file(file_path):
    """
    从 JSON 文件中加载字典。
    :param file_path: 字典保存文件的路径
    :return: 加载的字典
    """
    with open(file_path, 'r') as json_file:
        data_dict = json.load(json_file)
    print(f"Dictionary loaded from {file_path}")
    return data_dict

# 使用示例
if __name__ == "__main__":
    # 指定 result_msort 文件夹路径
    directory_path = '../result_story'  # 替换为实际路径
    output_file = 'msort_10_story.json'  # 保存字典的文件名

    # 加载所有子文件夹的 llmscore_modified.npy 数据到字典
    npy_data_dict = load_npy_files(directory_path)
    
    # 保存字典到 JSON 文件
    save_dict_to_file(npy_data_dict, output_file)
    
    # 从 JSON 文件重新加载字典
    loaded_dict = load_dict_from_file(output_file)
    
    # 验证加载结果
    print("Loaded Dictionary:", loaded_dict)