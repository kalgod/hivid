import os
import shutil

def remove_empty_directories(path):
    # 遍历文件夹
    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)
        
        # 检查文件夹是否为空
        if not os.listdir(folder_path):
            print(f"Removing empty directory: {folder_path}")
            os.rmdir(folder_path)

# 指定要检查的路径
directory_path = '../dataset/youtube'  # 请替换为你的目标目录路径
remove_empty_directories(directory_path)