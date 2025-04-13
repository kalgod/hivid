import math
import os
import numpy as np

def T(a):
    i = int(a * 2.0)
    # 使用近似比较来替代绝对误差
    if abs(a - 1) < 0.0001 or abs(a - 0.5) < 0.0001:
        return 1
    else:
        if i % 4 != 3:
            k = (i - (i % 4)) // 2
        else:
            k = (i + 1) // 2
        cha = a - k
        return T(k / 2) + T(k / 2 + cha) + 2 * k - 1 + 2 * cha

def main():
    m_list=[2,4,6,8,10,100]
    for m in m_list:
        num_list=[]
        len_list=[]
        video_list = os.listdir("../dataset/youtube")
        for (index, video_name) in enumerate(video_list): #enumerate函数同时遍历索引和目标，这里就是直接遍历video_list的每一项，index从0开始
            # print(f"Processing video {video_name} ({index + 1}/{len(video_list)})")
            video_folder = "../dataset/youtube"
            video_path = os.path.join(video_folder, video_name, f"{video_name}.mp4")   #指向.mp4文件
            frame_path = os.path.join(video_folder, video_name, "frames")    #指向该视频文件夹的frames文件夹
            gtscore_path = os.path.join(video_folder, video_name, "gtscore.npy")     #指向视频文件夹下的GT.npy
            info_path = os.path.join(video_folder, video_name, f"{video_name}_info.txt")    #指向info.txt
            
            gtscore = np.load(gtscore_path)
            global total_frame
            n = 125
            k = math.ceil((2 * n) / m) / 2
            i = 1.0
            res=T(k)
            num_list.append(res)
            len_list.append(n)
        avg_num=np.mean(num_list)
        avg_len=np.mean(len_list)
        print(f"m={m},avg_num={avg_num},avg_len={avg_len}")
        

if __name__ == "__main__":
    main()