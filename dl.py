import yt_dlp as youtube_dl
import os
import pandas as pd
from tqdm import tqdm
import h5py
import numpy as np
# import torch

def download_youtube_video(yt_id, video_id):
    try:
        youtube_url = f"https://www.youtube.com/watch?v={yt_id}"
        print(youtube_url)
        
        save_path = f"dataset/youtube/{video_id}"
        os.makedirs(save_path, exist_ok=True)  # 创建保存路径

        ydl_opts = {
            'outtmpl': os.path.join(save_path, f'{video_id}.mp4'),
            'format': 'bestvideo[height<=480]+bestaudio/best[height<=480]',  # 选择480p视频
            'cookiefile': 'youtube.txt',
            'merge_output_format': 'mp4',
        }

        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
        except Exception as download_e:
            print(f"Error downloading video {video_id}: {download_e}")
            return 0,[]

        video_file_path = os.path.join(save_path, f"{video_id}.mp4")
        
        print(f"successfully download video to {video_file_path}")
        
        title = info.get('title', 'Unknown Title')
        tags = info.get('tags', [])
        categories = info.get('categories', [])

        info_path = os.path.join(save_path, f"{video_id}_info.txt")
        
        try:
            with open(info_path, "w") as f:
                f.write(f"Video Title: {title}\n")
                f.write(f"Categories: {categories}\n")
                if tags:
                    f.write(f"Tags: {tags}\n")
        except Exception as open_e:
            print(f"Error writing info to {info_path}: {open_e}")
            return 0,[]

        print(f"successfully save video info to {info_path}")
        try:
            file_size = os.path.getsize(video_file_path) / (1024 * 1024)  # 获取文件大小
        except Exception as size_e:
            print(f"Error getting size of {video_file_path}: {size_e}")
            return 0,[]
        return file_size,categories

    except Exception as e:
        print(f"General error downloading video {video_id}: {e}")
        return 0,[]

size_sum = 0
size_max = 10

# 用于存储类别与对应ID的字典
category_dict = {}


# 获取用户输入的字符串
input_string = input("请输入vedio_id，用逗号隔开：")
# 使用split()方法分割字符串，并使用map()函数将分割后的字符串转换为整数列表
dllist = list(map(int, input_string.split(',')))


'''
video_list = ['video_25090', 'video_8530', 'video_7242', 'video_3192', 'video_16861', 'video_21572', 'video_28903', 'video_23163', 'video_6054', 'video_31491', 'video_11968', 'video_30442', 'video_27901', 'video_7263', 'video_6392', 'video_5687', 'video_28255', 'video_15803', 'video_12716', 'video_17691', 'video_25184', 'video_5404', 'video_21485', 'video_26318', 'video_5775', 'video_1245', 'video_18273', 'video_28155', 'video_9031', 'video_22106', 'video_23724', 'video_1828', 'video_29385', 'video_5645', 'video_11212', 'video_7232', 'video_18112', 'video_11628', 'video_11234', 'video_2562', 'video_28706', 'video_18513', 'video_1266', 'video_20025', 'video_4473', 'video_28359', 'video_28481', 'video_5405', 'video_6183', 'video_24127', 'video_2678', 'video_16751', 'video_13159', 'video_28347', 'video_4043', 'video_29101', 'video_21473', 'video_17011', 'video_13503', 'video_22233', 'video_19328', 'video_7014', 'video_25512', 'video_19002', 'video_7258', 'video_6867', 'video_29087', 'video_31401', 'video_3454', 'video_25154', 'video_11227', 'video_11183', 'video_19195', 'video_11181', 'video_23172', 'video_7209', 'video_29103', 'video_22307', 'video_26854', 'video_29885', 'video_3537', 'video_18018', 'video_6030', 'video_30757', 'video_21872', 'video_26851', 'video_21283', 'video_1102', 'video_18340', 'video_7026', 'video_5777', 'video_23967', 'video_29384', 'video_25816', 'video_3881', 'video_22500', 'video_1290', 'video_19332', 'video_18494', 'video_31260', 'video_13514', 'video_8037', 'video_1241', 'video_5437', 'video_21813', 'video_23695', 'video_8414', 'video_26521', 'video_20704', 'video_23931', 'video_23167', 'video_15557', 'video_496', 'video_6112', 'video_7474', 'video_5456', 'video_30553', 'video_5398', 'video_7436', 'video_3286', 'video_1149', 'video_28274', 'video_5875', 'video_16706', 'video_26935', 'video_29270', 'video_6235', 'video_10571', 'video_28861', 'video_375', 'video_3184', 'video_12293', 'video_29879', 'video_30971', 'video_3474', 'video_13700', 'video_13152', 'video_13577', 'video_23173', 'video_31529', 'video_6394', 'video_17697', 'video_17027', 'video_2468', 'video_5408', 'video_26590', 'video_362', 'video_6029', 'video_14484', 'video_19337', 'video_12709', 'video_26477', 'video_17877', 'video_25819', 'video_15807', 'video_23954', 'video_14024', 'video_25248', 'video_29803', 'video_1707', 'video_25829', 'video_4030', 'video_6175', 'video_4056', 'video_8518', 'video_26525', 'video_21096', 'video_5144', 'video_20456', 'video_8523', 'video_11080', 'video_14197', 'video_27967', 'video_12722', 'video_22619', 'video_23034', 'video_31485', 'video_23072', 'video_7356', 'video_6458', 'video_29734', 'video_28178', 'video_13966', 'video_4063', 'video_3526', 'video_10756', 'video_6813', 'video_25137', 'video_15821', 'video_23946', 'video_11969', 'video_6218', 'video_12992', 'video_13512', 'video_399', 'video_11201', 'video_30496', 'video_31465', 'video_20017', 'video_23059', 'video_23180', 'video_27892', 'video_14576', 'video_14001', 'video_2448', 'video_3883', 'video_17700', 'video_28860', 'video_31255', 'video_4475', 'video_31473', 'video_20090', 'video_1116', 'video_12893', 'video_20531', 'video_28281', 'video_2433', 'video_6110', 'video_398', 'video_24587', 'video_12290', 'video_15562', 'video_14032', 'video_921', 'video_30178', 'video_1231', 'video_504', 'video_17328', 'video_30478', 'video_23198', 'video_17098', 'video_28156', 'video_11244', 'video_29750', 'video_6844', 'video_29524', 'video_18543', 'video_6111', 'video_13470', 'video_11634', 'video_18310', 'video_31666', 'video_28471', 'video_5679', 'video_31253', 'video_29273', 'video_28256', 'video_26860', 'video_25188', 'video_14205', 'video_29806', 'video_14501', 'video_28157', 'video_12194', 'video_18351', 'video_24582', 'video_9666', 'video_6050', 'video_718', 'video_4051', 'video_25715', 'video_21808', 'video_17005', 'video_23444', 'video_8634', 'video_29810', 'video_20468', 'video_30500', 'video_17025', 'video_25207', 'video_31514', 'video_15556', 'video_18015', 'video_25158', 'video_29884', 'video_28950', 'video_1263', 'video_11245', 'video_6843', 'video_15543', 'video_25519', 'video_31271', 'video_15561', 'video_13962', 'video_1779', 'video_18838', 'video_2864', 'video_29096', 'video_284', 'video_21088', 'video_30514', 'video_20514', 'video_20186', 'video_31310', 'video_2564', 'video_1148', 'video_28160', 'video_30981', 'video_4477', 'video_25223', 'video_3933', 'video_16503', 'video_381', 'video_13697', 'video_3314', 'video_15454', 'video_31787', 'video_30237', 'video_1440', 'video_2861', 'video_13146', 'video_27096', 'video_2833', 'video_21285', 'video_26360', 'video_27535', 'video_7033', 'video_21576', 'video_21862', 'video_7458', 'video_2569', 'video_11959', 'video_5780', 'video_23057', 'video_18117', 'video_31657', 'video_5646', 'video_2439', 'video_1190', 'video_29744', 'video_28328', 'video_12899', 'video_11365', 'video_13495', 'video_18400', 'video_27020', 'video_20290', 'video_5393', 'video_25190', 'video_688', 'video_29816', 'video_17477', 'video_13961', 'video_8622', 'video_7358', 'video_28190', 'video_15818', 'video_23066', 'video_16335', 'video_11963', 'video_21810', 'video_19123', 'video_13042', 'video_13585', 'video_28478', 'video_19333', 'video_21556', 'video_13034', 'video_1239', 'video_15364', 'video_13167', 'video_6210', 'video_16221', 'video_15052', 'video_29105', 'video_22107', 'video_22223', 'video_23182', 'video_9656', 'video_23959', 'video_30563', 'video_30566', 'video_5237', 'video_22785', 'video_28939', 'video_20517', 'video_6221', 'video_2441', 'video_3878', 'video_6840', 'video_3345', 'video_27970', 'video_7352', 'video_19001', 'video_27007', 'video_27542', 'video_31268', 'video_31390', 'video_29388', 'video_1714', 'video_31513', 'video_8038']
print(len(video_list))
# 提取数字并转换为整数
dllist = [int(video.split('_')[1]) for video in video_list]
'''

meta_data = "dataset/metadata.csv"
dataset = 'dataset/mr_hisum.h5'
video_data = h5py.File(dataset, 'r')
df = pd.read_csv(meta_data)
df_ls=list(df.itertuples())  #备注df_ls[0] = "Pandas(Index=0, video_id='video_1', yt8m_file='train0026', random_id='ORaA', youtube_id='JhdjUam0l6A', duration=258, views=84554, labels='[8]')"

selected_df_ls = []
for id in dllist:
    selected_df_ls.append(df_ls[id-1])

for row in tqdm(selected_df_ls):
    yt_id = row.youtube_id
    video_id = row.video_id
    size,categories = download_youtube_video(yt_id, video_id)

    if size != 0:
        gtscore = np.array(video_data[video_id + '/gtscore'])
        n_frames = gtscore.shape[0]
        for category in categories:
            if category not in category_dict:
                category_dict[category] = []
            category_dict[category].append(video_id)
        np.save(os.path.join("dataset/youtube", video_id, "gtscore.npy"), gtscore)
        np.save(os.path.join("dataset/youtube", video_id, "n_frames.npy"), n_frames)
        print(f"successfully save gtscore and n_frames to {video_id}")
    size_sum = size_sum + size
    print(f"-{size_sum}-")
    if size_sum >= size_max * 1024:
        print(f'Total downloaded size exceeds {size_max}GB limit. Stopping download.')
        break

# 写入 TXT 文件
with open("categories.txt", "w") as f:
    for category, ids in category_dict.items():
        f.write(f"{category}: {', '.join(map(str, ids))}\n")

print("数据已成功写入到 categories.txt 文件中。")
print("finish")