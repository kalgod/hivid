import os
import cv2
import argparse
from poe_api_wrapper import PoeApi
import numpy as np
from scipy.stats import mode
import csv
import shutil
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize
from time import sleep
from collections import defaultdict
import matplotlib.pyplot as plt
import random

bot_number = 0
total_frame = 0

# Tokens for Poe API (update with actual tokens if needed)
tokens = {
    'p-b': "i7JZq6B4feBIrYx2Ao8ewQ==", 
    'p-lat': "cuHNeEA7iTTHDGTzlgzdCa1b6niVntYU6vSQmOqr5g==",
    'formkey': 'c045ce1e4e5ba81ffa20da079984c515',
}

tokens_jlc = {
    'p-b': "iG6CkxM-wGYwbqiYIkFDrA==", 
    'p-lat': "moQLC5Bz/NR66fad6mHHR8kQ9Vtecv492PW5a8c6AQ==",
    'formkey': 'c8ceeb7e8b2c67b13aceb888ddbfa20d',
}

#得到剩余的llm积分
def get_data(client):
    # Get chat data of all bots (this will fetch all available threads)
    # print(client.get_chat_history("claude-3.5-sonnet")['data'])
    # Get chat data of a bot (this will fetch all available threads)
    # print(client.get_chat_history())
    data = client.get_settings()
    print(data["messagePointInfo"]["messagePointBalance"])
    # print(client.get_available_creation_models())
    # print(client.get_available_bots())

#eval_llm使用 
def send_message(image_path, client,info):
    global total_frame

    # 以下是第一个图片的描述，给了图片的数量和视频的信息
    message = f"""
    I have uploaded {len(image_path)} frames, each representing a video chunk of 1 seconds. You first extract the frame number attached below the image content. The original video informations are{info}
    Your task is as below:
    1. Based on the video information background, first summarize some keywords that may attract viewers. Then based on your keywords, analyze each image content.
    2. Based on your analysis, on a scale of integer (0,100), rate all the {len(image_path)} frames that higher number means higher interestingness score. Different frames can yield same scores.
    Remember: There are {total_frame} frames in total. Usually viewers are not interested in frames around the end of the vedio, and have interests at the beginning of the vedio, although those frames are meaningless.
    According to this, you have to give frame{total_frame-1} a rating of 0, and frame0 a rating near 50. Frames similar to those two will have similar ratings.

    Your answer must be a json format like this: 
    ```json
    [
        ("frame": xxx, "rating": xxx),
        ("frame": xxx, "rating": xxx),
        ("frame": xxx, "rating": xxx)
    ]
    ```
    You must show with frame number ascending. Below your json answer, analyze each image content and exaplain your rating.
    """
    



    frame = ["temp2.png"]
    if image_path is not None:
        frame = image_path

    # bot = "gpt4_o";chatId= 662480034
    # bot = "gpt4_o_128k";chatId= 649712483
    # bot = "gpt4_o_mini";chatId= 649104377
    # bot="gpt4_o_mini_128k";chatId=667358994
    # bot="claude-3.5-sonnet";chatId=685220083
    # bot="claude_3_opus";chatId=None
    # bot = "gpt4_o_128k";chatId= None
    
    global bot_number

    if bot_number == 1:
        bot="gpt4_o_mini_128k";chatId=667358994
        bot_number = 0
    elif bot_number == 0:
        bot="gpt4_o_mini_128k";chatId=667360175
        bot_number = 1
    else:
        raise bot_number_error

    for chunk in client.send_message(bot=bot, message=message,file_path=frame,chatId= chatId): pass #一次性发送了信息和path中的所有图片
    res = chunk["text"] #从返回的chunk中取出了llm的返回值
    client.chat_break(bot=bot,chatId=chatId)
    return res


#给视频分块  暂时不看了
def extract_chunks(video_path, chunk_duration, frame_path):
    # 从视频中提取帧，每个帧代表一个视频块
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
    os.system("rm -rf " + frame_path+"/*")
    fps=np.ceil(fps)
    print(f"{video_path} Video FPS: {fps}")
    frames_per_chunk = int(fps * chunk_duration)
    res = []

    while True:
        chunk_frames = []
        for _ in range(frames_per_chunk):
            success, frame = video.read()
            if not success:
                break
            chunk_frames.append(frame)
        if not chunk_frames:
            break
        if (len(chunk_frames) < frames_per_chunk):
            break
        # 临时策略：只取每个 chunk 的第一帧
        key_frame1 = chunk_frames[0]
        key_frame2 = chunk_frames[frames_per_chunk//2]
        res.append(key_frame1)
        res.append(key_frame2)
    if len(chunk_frames)>1:
        res.append(chunk_frames[-2])
        res.append(chunk_frames[-1])
    video.release()
    print(f"Extracted {len(res)} chunks from the video.")
    os.makedirs(frame_path, exist_ok=True)
    for i in range(len(res)):
        # cv2.imwrite(os.path.join(frame_path, f"{i}.png"), res[i])
        modify_image(res[i],f"frame: {i}",os.path.join(frame_path, f"{i}.png"))
    return len(res)

#eval_llm使用，解析LLM返回的JSON字符串                                                                                                                                             
def extract_score(res):
    # if ("json" not in res): return 0,None
    if ("[" not in res): return 0,None
    # if ("I'm unable to" in res or "unable to view" in res): return 0,None
    start_index = res.index('[')
    end_index = res.index(']') + 1
    # print(f"Start index: {start_index}, End index: {end_index}")

    # 提取 JSON 字符串
    json_str = res[start_index:end_index]
    # print(f"JSON string: {json_str}")
    frame=[]
    rating=[]
    # 将字符串按行分割，并遍历每一行
    for line in json_str.splitlines():
        line = line.strip()  # 去掉前后空白
        # print(line)
        if "{" in line and "}" in line:  # 确保是一个有效的字典行
            # 提取 frame 和 rating
            frame_part = line.split('"frame":')[1].split(',')[0].strip()
            rating_part = line.split('"rating":')[1].split('}')[0].strip()
            
            # 打印结果
            frame_ = int(frame_part)
            rating_ = float(rating_part)
            rating_ = int(rating_)
            frame.append(frame_)
            rating.append(rating_)
            # print(f"line Frame: {frame_}, Rating: {rating_}")
    frame=np.array(frame)
    rating=np.array(rating)
    sorted_indices = np.argsort(frame)
    rating=rating[sorted_indices]
    frame=frame[sorted_indices]
    return 1,frame,rating  #返回frame升序的frame，rating

#建立一个图片（可能有解释）
def modify_image(image,text,image_path):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # 白色
    thickness = 2
    line_type = cv2.LINE_AA

    # 计算文本大小
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # 创建一个新的图像，黑色背景
    new_image = cv2.resize(image, (image.shape[1], image.shape[0] + text_height + baseline))
    new_image[:] = (0, 0, 0)  # 填充黑色背景
    new_image[0:image.shape[0], 0:image.shape[1]] = image  # 添加原图

    # 在新图像底部添加文本
    text_x = int((new_image.shape[1] - text_width) / 2)  # 水平居中
    text_y = image.shape[0] + text_height + baseline - 10  # 垂直位置，向上偏移一点
    cv2.putText(new_image, text, (text_x, text_y), font, font_scale, font_color, thickness, line_type)
    cv2.imwrite(image_path, new_image)

#输入一个评分列表，返回对应的排名列表，最小为1
def convert_to_index(score):
    array=score
    unique_sorted = sorted(set(array))
    # 创建一个映射，将浮点数映射到它们的相对排名
    rank_mapping = {value: index + 1 for index, value in enumerate(unique_sorted)}
    # 使用映射转换原数组
    ranked_array = [rank_mapping[num] for num in array]
    return np.array(ranked_array)

#把num个数分成n个一组一组的，从0开始
def create_groups(num, n):
    groups = []
    for i in range(0, num, n):
        group = list(range(i, min(i + n, num)))  # 创建一个组，范围是 [i, i+n)
        groups.append(group)
    return groups

#选了n-2个满足正态分布的值，作为参考
def select_score_indices(score, n):
    if n < 2:
        raise ValueError("n must be at least 2 to include both min and max.")
    
    # 计算最低和最高值及其索引
    min_score = min(score)
    max_score = max(score)
    
    min_index = score.index(min_score)
    max_index = score.index(max_score)
    
    # 剩余选项，排除最低和最高
    remaining_indices = [i for i in range(len(score)) if i!=min_index and i!=max_index]
    
    # 确保剩余的值足够选取
    if len(remaining_indices) < (n - 2):
        raise ValueError("Not enough scores to select from.")
    
    # 计算均值和标准差
    mean = np.mean([score[i] for i in remaining_indices])
    std_dev = np.std([score[i] for i in remaining_indices])
    
    # 随机选择正态分布的值
    normal_samples = np.random.normal(mean, std_dev, n - 2)
    
    # 限制选取的数在有效范围内
    normal_samples = np.clip(normal_samples, min_score, max_score)
    
    # 将随机选取的数索引找出
    sampled_indices = set()
    for sample in normal_samples:
        closest_index = (np.abs(np.array([score[i] for i in remaining_indices]) - sample)).argmin()
        sampled_indices.add(remaining_indices[closest_index])
    
    # 如果样本不足，随机再选取一些值
    while len(sampled_indices) < (n - 2):
        extra_index = np.random.choice(remaining_indices)
        sampled_indices.add(extra_index)
        
    # 组合结果并排序
    final_indices = sorted(list(sampled_indices) + [min_index, max_index])
    
    return final_indices

def eval_llm(gt_scores,info,frame_path,client,eval_path): #gt_scores是npy数组，info是文本，frame文件夹路径，client，eval.txt路径
    frame_num=gt_scores.shape[0]*1
    pre_frame_path=os.path.dirname(frame_path)
    path_list=os.listdir(frame_path)
    assert len(path_list)==frame_num
    tmp_path=pre_frame_path+"/tmp"
    os.makedirs(tmp_path, exist_ok=True)
    image_path=[]
    eval_frame_num=4  #4个一组
    final_score=[]
    last_res=""
    groups=create_groups(frame_num,eval_frame_num)  #把所有的帧分成4个一组
    for i in range(len(groups)): #遍历所有的组
        image_path=[]
        cur_group=groups[i] #目前的组
        if (i!=0): #不是第一组
            tmp_idx=select_score_indices(final_score,2*eval_frame_num-len(cur_group)) #前面是最终结果列表，后者就是4
            # last_group=groups[i-1]
            for j in range(len(tmp_idx)):
                cur_path = os.path.join(tmp_path, f"{tmp_idx[j]}.png") #加入了参考图片
                image_path.append(cur_path)
        
        for j in range(len(cur_group)): #在当前这一组中，遍历1234，给image_path里面输入这些地址，命名就是0123456789………………
            cur_path = os.path.join(frame_path, f"{cur_group[j]}.png")
            image_path.append(cur_path)

        print(i,image_path) #打印组号，还有他们的路径
        for j in range(5): #尝试五次llm请求
            res = send_message(image_path=image_path, client=client,info=info) #image_path里面有4个图片的路径
            print(res)
            check,llm_score=extract_score(res) #默认check返回1，score返回的是score数组
            if (check==1 and res!=last_res and len(llm_score)==len(image_path)): break #合法的返回值
            else:
                print("No json or wrong score length, repeat!!!!!!!!!!!!!!!!!!!!!!!!!!")
                sleep(5)
        if (check==0): 
            print("Error")
            exit(0)
        last_res=res
        # print(llm_score)
        print("\n###################################################\n###################################################\n")
        #完成了这个组！

        cur_group=groups[i]
        for j in range(len(cur_group)):
            cur_path = os.path.join(frame_path, f"{cur_group[j]}.png")
            image_path.append(cur_path)
            #把frame里面的图片加上rating，放到tmp文件夹
            modify_image(cv2.imread(cur_path),f"Rating: {llm_score[j-len(cur_group)]}",os.path.join(tmp_path, f"{cur_group[j]}.png"))
            final_score.append(llm_score[j-len(cur_group)])
        sleep(5)
    final_score=average_every_n_elements(final_score,1)
    final_score=np.array(final_score)
    idx_gt_scores=convert_to_index(gt_scores)
    idx_final_scores=convert_to_index(final_score)

    #已经全部处理完毕，输出llmscore和gtscore
    print("LLM score:",final_score,"\nGT score:",gt_scores)

    # print("fenduan")
    # for i in range(len(groups)):
    #     cur_group=groups[i]
    #     cur_gt_score=gt_scores[cur_group]
    #     cur_final_score=final_score[cur_group]
    #     print(f"GT: {cur_gt_score} Final: {cur_final_score}")
    #     plcc, plcc_p_value = pearsonr(cur_final_score, cur_gt_score)
    #     plcc_all.append(plcc)
    #     srcc, srcc_p_value = spearmanr(cur_final_score, cur_gt_score)
    #     srcc_all.append(srcc)
    #     print(f"(PLCC): {plcc}, p-value: {plcc_p_value}")
    #     print(f"(SRCC): {srcc}, p-value: {srcc_p_value}")
    # print("Average of fenduan",np.mean(plcc_all),np.mean(srcc_all))

    print("ALL PLCC")
    plcc, plcc_p_value = pearsonr(final_score, gt_scores)
    srcc, srcc_p_value = spearmanr(final_score, gt_scores)
    plcc_idx, plcc_idx_p_value = pearsonr(idx_final_scores, idx_gt_scores)
    srcc_idx, srcc_idx_p_value = spearmanr(idx_final_scores, idx_gt_scores)
    
    outlog = f"""
    (PLCC): {plcc}, p-value: {plcc_p_value}, Index PLCC: {plcc_idx}, p-value: {plcc_idx_p_value}
    (SRCC): {srcc}, p-value: {srcc_p_value}, Index SRCC: {srcc_idx}, p-value: {srcc_idx_p_value}
    """
    print(outlog)

    with open(eval_path, "w") as f:
        f.write(outlog)
    return plcc,srcc,plcc_idx,srcc_idx,final_score

#每n个数求一个平均值
def average_every_n_elements(a, n=4):
    # a=convert_to_index(a)
    averages = []
    for i in range(0, len(a), n):
        chunk = a[i:i+n]
        avg = np.mean(chunk)
        averages.append(avg)
    averages=np.array(averages)
    # averages=averages/np.sum(averages)
    return averages

#未见调用
def read_mos(filename):
    content_dict = defaultdict(list)
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            streaming_log = row['streaming_log']
            mos = float(row['mos'])  
            content = row['content']
            content_dict[content].append([streaming_log, mos])
    return content_dict

#eval_qoe()有调用，但该函数未被调用
def read_qos(filename):
    res=[]
    with open("./sqoe3/mos/streaming_logs/"+filename, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            vmaf=float(row['vmaf'])
            rebuffer=float(row['rebuffering_duration'])
            bitrate=float(row['video_bitrate'])
            res.append([vmaf,rebuffer,bitrate/1000.0])
    return np.array(res)

#eval_qoe()有调用，但该函数未被调用
def compute_qoe_comyco(qos):
    qoe=[]
    for i in range (len(qos)):
        chunk_qoe=0
        vmaf=qos[i][0]
        rebuffer=qos[i][1]
        chunk_qoe=chunk_qoe+0.8469*vmaf-28.7959*rebuffer
        if (i<len(qos)-1):
            pos_switch=max(0,qos[i+1][0]-vmaf)
            neg_switch=max(0,vmaf-qos[i+1][0])
            chunk_qoe=chunk_qoe+0.2979*pos_switch-1.0610*neg_switch
        qoe.append(chunk_qoe)
    qoe=np.array(qoe)
    return qoe

#未见调用
def compute_qoe_mpc(qos):
    qoe=[]
    for i in range (len(qos)):
        chunk_qoe=0
        vmaf=qos[i][0]
        rebuffer=qos[i][1]
        bitrate=qos[i][2]
        chunk_qoe=chunk_qoe+bitrate-7.0*rebuffer
        if (i<len(qos)-1):
            switch=np.abs(qos[i+1][2]-bitrate)
            chunk_qoe=chunk_qoe-switch
        qoe.append(chunk_qoe)
    qoe=np.array(qoe)
    return qoe

#存一个 NumPy 数组
def save_weight(path,weight):
    weight=np.array(weight)
    np.save(path,weight)

#用于加载存储在磁盘上的 NumPy 数组或其他数据（例如存储为 .npy 或 .npz 格式）的文件
def load_weight(path):
    weight=np.load(path)
    return weight

#eval_qoe()有调用（已注释），但该函数未被调用
def optimal_weight(qoe,mos):
    # 优化权重
    chunk_num=qoe.shape[0]
    def objective(a):
        c = np.dot(a, qoe)  # 计算 c
        correlation, _ = pearsonr(c, mos)  # 计算 SRCC
        return -correlation#+0.1*np.mean(np.maximum(0,-a))  # 负值，因为我们要最大化
    opt=np.ones(chunk_num)
    bounds = [(0.5, 1) for _ in range(chunk_num)]
    result = minimize(objective, opt, bounds=bounds)
    optimal_a = result.x
    optimal_a=optimal_a/np.sum(optimal_a)
    print("Optimal a:", optimal_a)
    return optimal_a

#未见调用
def eval_qoe(weight,name,mos_all):
    mos_list=mos_all[name]
    mos_gt=[float(row[1]) for row in mos_list]
    mos_gt=np.array(mos_gt)
    print("#####Evaluating QoE Now\n Weight:",weight,"MOS:",mos_list)
    qoe_all=[]
    for i in range(len(mos_list)):
        filename=mos_list[i][0]
        qos=read_qos(filename)
        qoe=compute_qoe_comyco(qos)
        qoe_all.append(qoe)
    qoe_all=np.array(qoe_all)
    # weight=optimal_weight(qoe_all.T,mos_gt)
    # weight=[2.2397,2.2687,2.1694,2.1138,2.1065]
    # weight=np.array(weight)
    # weight=weight/np.sum(weight)

    mos_ori=np.sum(qoe_all,axis=1)
    plcc_ori, plcc_p_value = pearsonr(mos_ori, mos_gt)
    srcc_ori, srcc_p_value = spearmanr(mos_ori, mos_gt)
    print(f"Original (PLCC): {plcc_ori} \t (SRCC): {srcc_ori}")

    mos_llm=np.dot(weight,qoe_all.T)
    plcc_llm, plcc_p_value = pearsonr(mos_llm, mos_gt)
    srcc_llm, srcc_p_value = spearmanr(mos_llm, mos_gt)
    print(f"LLM (PLCC): {plcc_llm} \t (SRCC): {srcc_llm}\n")
    return plcc_ori,srcc_ori,plcc_llm,srcc_llm

#未见调用
def plot_cdf(array1, array2,title,metric):
    labels=('Original', 'Weighted')
    # 计算 CDF
    def calculate_cdf(data):
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(data) + 1) / len(data)
        return sorted_data, cdf

    sorted_array1, cdf_array1 = calculate_cdf(array1)
    sorted_array2, cdf_array2 = calculate_cdf(array2)

    # 绘制 CDF 图
    plt.figure(figsize=(12, 8))
    plt.plot(sorted_array1, cdf_array1, label=labels[0], color='blue')
    plt.plot(sorted_array2, cdf_array2, label=labels[1], color='orange')
    plt.title(title,fontsize=25)
    plt.xlabel(metric,fontsize=25)
    plt.ylabel('CDF',fontsize=25)
    plt.legend(fontsize=25)
    plt.tick_params(axis='both', labelsize=25)
    plt.grid(True)
    plt.savefig("./figs/"+title+"_"+metric+".png")

#绘图
def plot_score(score1, score2, label1, label2, title, save_path):
    if len(score1) != len(score2):
        print("The two scores must have the same length")
        return
    score1_norm = np.array(score1)
    score2_norm = np.array(score2) # / 100  打分就除以100
    x = range(len(score1))

    plt.plot(x, score1_norm, color='blue', label=label1)
    plt.plot(x, score2_norm, color='red', label=label2)

    plt.legend()
    plt.xlabel('Frame Index')
    plt.ylabel('Normalized Scores')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    return save_path



def eval_llm_msort(gt_scores,info,frame_path,client,eval_path,m): #gt_scores是npy数组，info是文本，frame文件夹路径，client，eval.txt路径

    #以下建立了temp文件夹(路径为temp_path)，用于存储带有分数的frame
    frame_num=gt_scores.shape[0]*1  #理论上应该有的frames
    pre_frame_path=os.path.dirname(frame_path)  #指向该视频文件夹的frames文件夹的上一级
    path_list=os.listdir(frame_path)
    assert len(path_list)==frame_num  #assert用于判断两者是否相同，此处仅用于排除错误
    tmp_path=pre_frame_path+"/tmp"
    os.makedirs(tmp_path, exist_ok=True)    #temp_path 存储了带有分数的frame

    ### end chunk

    path_list = []
    for i in range(len(os.listdir(frame_path))):
        path_list.append(os.path.join(frame_path, f"{i}.png"))


    ###给path_list分组，存到groups数组
    groups = []
    temp = []
    for i in range(len(path_list)):
        if i != 0 and i % m == 0 :
            groups.append(temp)
            temp = []
        if i == len(path_list)-1:
            temp.append(path_list[i])
            groups.append(temp)
            break
        temp.append(path_list[i])

    ### end chunk
    #把所有的帧分成m个一组

    result = merge_sort(groups,m,client,info)     #返回了一个数组，[path1,path2^^^^^^]按重要性从高到低排列

    rank_list = []
    for i in range(len(result)):
        #把frame里面的图片加上rating，放到tmp文件夹
        cur_path = result[i]
        modify_image(cv2.imread(cur_path),f"Ranking: {i}",os.path.join(tmp_path, f"{os.path.basename(cur_path)}"))
        rank_list.append(os.path.basename(cur_path))

    #rank_list存储的全都是照片的名字（'0.png'），转换成数字
    number_list = [int(filename.split('.')[0]) for filename in rank_list]

    print(number_list)

    final_score = [-1] * len(number_list)
    for index,name in enumerate(number_list):
        final_score[name] = float((len(number_list)-1-index)/(len(number_list)-1))


    final_score=np.array(final_score)
    idx_gt_scores=convert_to_index(gt_scores)
    idx_final_scores=convert_to_index(final_score)

    #已经全部处理完毕，输出llmscore和gtscore
    print("LLM score:",final_score,"\nGT score:",gt_scores)

    # print("fenduan")
    # for i in range(len(groups)):
    #     cur_group=groups[i]
    #     cur_gt_score=gt_scores[cur_group]
    #     cur_final_score=final_score[cur_group]
    #     print(f"GT: {cur_gt_score} Final: {cur_final_score}")
    #     plcc, plcc_p_value = pearsonr(cur_final_score, cur_gt_score)
    #     plcc_all.append(plcc)
    #     srcc, srcc_p_value = spearmanr(cur_final_score, cur_gt_score)
    #     srcc_all.append(srcc)
    #     print(f"(PLCC): {plcc}, p-value: {plcc_p_value}")
    #     print(f"(SRCC): {srcc}, p-value: {srcc_p_value}")
    # print("Average of fenduan",np.mean(plcc_all),np.mean(srcc_all))

    print("ALL PLCC")
    plcc, plcc_p_value = pearsonr(final_score, gt_scores)
    srcc, srcc_p_value = spearmanr(final_score, gt_scores)
    plcc_idx, plcc_idx_p_value = pearsonr(idx_final_scores, idx_gt_scores)
    srcc_idx, srcc_idx_p_value = spearmanr(idx_final_scores, idx_gt_scores)
    
    outlog = f"""
    (PLCC): {plcc}, p-value: {plcc_p_value}, Index PLCC: {plcc_idx}, p-value: {plcc_idx_p_value}
    (SRCC): {srcc}, p-value: {srcc_p_value}, Index SRCC: {srcc_idx}, p-value: {srcc_idx_p_value}
    """
    print(outlog)

    with open(eval_path, "w") as f:
        f.write(outlog)
    return plcc,srcc,plcc_idx,srcc_idx,final_score

def merge_sort(arr,m,client,info):
    # 基本情况：如果数组长度小于或等于1，则返回数组
    if len(arr) == 1:
        return merge(m,arr[0],client,info)

    # 找到数组的中间索引
    mid = len(arr) // 2

    # 递归地对左右两半进行归并排序
    left_half = merge_sort(arr[:mid],m,client,info)
    right_half = merge_sort(arr[mid:],m,client,info)

    # 合并已排序的左右两半
    return merge(m,left_half, right_half,client,info)

def merge(*args):  #承载了排序和合并两大功能    (m,1,client,info)或者(m,1,2,client,info).  注意这里过来的都是[1,2,3,4],但返回是[[]]
    path = os.path.dirname(args[1][0])
    merged = [[]]
    if len(args) == 4:  #只有一个groups，直接排序输出
        m = args[0]
        client = args[2]
        info = args[3]
        if len(args[1]) <= m:
            eval_path = args[1]
            print("只有一项，而且个数不大于m")
            position = eval_llm_m(eval_path,client,info)
            for i in position:
                merged[0].append(os.path.join(path, f"{i}.png"))

        else:
            print("排序过了直接输出",args[1])
            return [args[1]]  #大于m的代表已经排序过了，直接返回即可

    if len(args) == 5:  #两个groups
        m,remain1,remain2,client,info = args
        total = len(args[1]) + len(args[2])
        while len(merged[0]) != total:
            if len(remain1)*len(remain2) == 0:
                merged[0] = merged[0]+remain1+remain2
            elif len(remain1)+len(remain2) > m:
                eval_path = remain1[:m//2]+remain2[:m//2]
                print(len(remain1),len(remain2),len(eval_path),"正常两组项")
                position = eval_llm_m(eval_path,client,info)
                for i in range(m//2):
                    merged[0].append(os.path.join(path, f"{position[i]}.png"))
                    #merged[0].append(eval_path[position[i]])
                    if os.path.join(path, f"{position[i]}.png") in remain1:
                        remain1.remove(os.path.join(path, f"{position[i]}.png"))
                    else:
                        remain2.remove(os.path.join(path, f"{position[i]}.png"))
            else:
                eval_path = remain1+remain2
                print(len(remain1),len(remain2),len(eval_path),"两组剩余项")
                position = eval_llm_m(eval_path,client,info)
                for i in range(len(eval_path)):
                    merged[0].append(os.path.join(path, f"{position[i]}.png"))
                    #merged[0].append(eval_path[position[i]])

    return merged[0]

def eval_llm_m(image_path,client,info):

    name_set = set()
    for i in image_path:
        name_set.add(os.path.basename(i))

    last_res=""
    for j in range(5): #尝试五次llm请求
        res = send_message(image_path=image_path, client=client,info=info) #image_path里面有4个图片的路径
        print(res)
        check,frame,llm_score=extract_score(res) #默认check返回1，frame返回了输出的帧的数字列表，score返回的是score数组

        frame = frame.tolist()
        llm_score = llm_score.tolist()

        frame_name = set()
        frame_name = set( [str(i)+".png" for i in frame] )
        if (check==1 and res!=last_res and len(llm_score)==len(image_path) and len(frame) == len(image_path) and  frame_name == name_set): #len(llm_score) == len(set(llm_score)) and
            print("合法",frame,"$$$",llm_score)
            break #合法的返回值,   #无重复元素
        else:
            print("No json or wrong score length, repeat!!!!!!!!!!!!!!!!!!!!!!!!!!")
            #sleep(5)
        if (check==0): 
            print("Error")
            exit(0)
        last_res=res
        # print(llm_score)
        print("\n###################################################\n###################################################\n")
        #完成了这个组！

    '''
    for j in range(5): #尝试五次llm请求
        res = send_message(image_path=image_path, client=client,info=info) #image_path里面有m个图片的路径
        print(res)
        check,llm_rating=extract_score(res) #默认check返回1，llm_rating返回的是重要性排序数组
        if (check==1 and res!=last_res and len(llm_rating)==len(image_path)): break #合法的返回值 np.array_equal(llm_rating, np.array([0, 1, 2, 3, 4, 5, 6, 7]))
        else:
            print("No json or wrong score length, repeat!!!!!!!!!!!!!!!!!!!!!!!!!!")
            sleep(5)
    if (check==0): 
        print("Error")
        exit(0)
    
    
    ######模拟llm结果，全部正序
    llm_score = []
    for i in range(len(image_path)):
        llm_score.append(i)
    
    print(llm_score, image_path)
    '''
    
    # 将 frame 和 rating 组合成一个元组列表
    combined = list(zip(frame, llm_score))

    # 根据 rating（降序）和 frame（升序）进行排序
    sorted_combined = sorted(combined, key=lambda x: (-x[1], x[0]))

    # 提取排序后的 frame 名称
    sorted_frames = [frame for frame, rating in sorted_combined]

    '''
    llm_score = [-i for i in llm_score]
    llm_score = rank_array(llm_score)
    positions = [-1] * len(image_path)  # 初始化为 -1
    for index,rank in enumerate(llm_score):
        while(positions[rank] != -1):
            rank = rank+1
        positions[rank] = frame[index]
    '''
    
    '''
    # 初始化一个固定大小的数组用来存储位置
    positions = [-1] * len(image_path)  # 初始化为 -1，表示未找到该值的位置
    llm_score = rank_array(llm_score) #分数转为0123456---
    number_list = []
    for i in image_path:
        number_list.append(os.path.basename(i))
    number_list = [int(filename.split('.')[0]) for filename in number_list]

    number_arr = rank_array(number_list)   #标志了每个帧在numberlist中的位置
    llm_score = rank_array(llm_score)   #转化为分数（排名）
    real_score = [-1] * len(number_arr)  #按上传顺序的分数（0123）
    for i in range(len((number_arr))):
        real_score[i] = llm_score[number_arr[i]]
    for index,score in enumerate(real_score):
        positions[len(number_arr)-score-1] = index
    '''

    print("排序",sorted_frames)
    #raise e
    '''
    #llm分数越高，这里分数越低
    for i in range(len(llm_score)):
        llm_score[i] = -llm_score[i]

    #分数最低的，给评分0（llm最高，给0）
    llm_score = rank_array(llm_score)

    # 初始化一个固定大小的数组用来存储位置
    positions = [-1] * len(image_path)  # 初始化为 -1，表示未找到该值的位置

    # 遍历原始数组，记录每个值的位置
    for index, value in enumerate(llm_score):
        positions[value] = index
    '''

    #sleep(5)
    return(sorted_frames)   #尽管gpt是乱序的，我们依然能找到……吧，直接返回图片的序号，重要性递减，eg[1,5,4,3,52,100]
    
#对于一个列表（数字越大优先级越高，返回排名值，最小的获得0）
def rank_array(arr):

    arr = np.array(arr)
    
    # 对数组进行排序，同时获取原始索引
    sorted_indices = np.argsort(arr)
    sorted_arr = arr[sorted_indices]

    # 初始化排名数组
    ranks = np.empty_like(sorted_indices)

    # 遍历排序后的数组，分配排名
    current_rank = -1
    for i in range(len(sorted_arr)):
        # 如果不是第一个元素且当前元素与前一个元素不同，则增加排名
        if i > 0 and sorted_arr[i] == sorted_arr[i - 1]:
            ranks[sorted_indices[i]] = ranks[sorted_indices[i]-1]
            current_rank +=1
            continue
        current_rank += 1
        ranks[sorted_indices[i]] = current_rank

    #print("to list", ranks.tolist())
    return ranks.tolist()  # 排名从0开始，数越大，排名越高

def main():
    #以下是一些初始变量，从命令行提取，查看help可以用 python qoe.py --help
    #不使用初始值时，用 python qoe.py -dataset xxxx(./dataset/下面的文件夹名) -chunk xxxx(秒数)
    parser = argparse.ArgumentParser(description="Extract frames from videos and process them.")  
    parser.add_argument("-dataset",type=str, default="youtube", help="Folder containing videos to process.")
    parser.add_argument("-chunk", type=float,default=1, help="Duration of each chunk in seconds.")
    args = parser.parse_args()

    #在这里定义了视频的文件夹和每个chunk的秒数
    video_folder = "./dataset/"+args.dataset
    chunk_duration = args.chunk

    #猜测是链接llm的账户api
    client = PoeApi(tokens=tokens_jlc)  # Adjust this to use the correct tokens if necessary
    get_data(client=client)
    
    #这里是要分析的视频
    # video_list = os.listdir(video_folder)
    video_list = ["video_115"]
    # video_list = ["video_112","video_115","video_117","video_118","video_136","video_15","video_24","video_35","video_61"]
    # plcc_ori_all=[]
    # srcc_ori_all=[]
    # plcc_llm_all=[]
    # srcc_llm_all=[]


    for (index, video_name) in enumerate(video_list): #enumerate函数同时遍历索引和目标，这里就是直接遍历video_list的每一项，index从0开始
        # if index in [0, 1, 2, 3, 4, 5]: continue

        #生成通用路径

        video_path = os.path.join(video_folder, video_name, f"{video_name}.mp4")   #指向.mp4文件
        frame_path = os.path.join(video_folder, video_name, "frames")    #指向该视频文件夹的frames文件夹
        gtscore_path = os.path.join(video_folder, video_name, "gtscore.npy")     #指向视频文件夹下的GT.npy
        info_path = os.path.join(video_folder, video_name, f"{video_name}_info.txt")    #指向info.txt

        result_path = os.path.join('./result_new',video_name) #结果存在本目录下的result_new/video_name文件夹下
        if not os.path.exists(result_path): #没有就创造directory
            os.makedirs(result_path)
        llm_score_path = os.path.join(result_path, "llmscore.npy") #把llm打分的存储路径存成llmscore.npy
        eval_path = os.path.join(result_path, "eval.txt") #eval.txt

        #创造gt数组，然后输出gt的个数
        gtscore = load_weight(gtscore_path)
        print("num of gtscore:",gtscore.shape[0])

        global total_frame
        total_frame = gtscore.shape[0]

        # 第一次运行时已提取，故不再重复进行
        # chunk_num = extract_chunks(video_path=video_path, chunk_duration=chunk_duration, frame_path=frame_path)
        # print(f"extract {chunk_num} frames of {video_name}")

        #输出正在分析的视频的信息
        with open(info_path, 'r') as f:
            info = f.read()
        print(f"Processing {video_name}\n{info}")
        
        #m=int(input("输入m:"))
        #plcc,srcc,plcc_idx,srcc_idx,weight =  eval_llm_msort(gtscore,info,frame_path,client,eval_path,m)  #除了weight以外都是数，weight是得分数组

        number_list = [95, 96, 94, 93, 100, 99, 98, 97, 82, 81, 80, 79, 40, 41, 46, 47, 76, 39, 38, 73, 92, 72, 64, 63, 65, 71, 60, 57, 19, 26, 27, 48, 101, 103, 59, 70, 28, 25, 24, 20, 58, 18, 17, 23, 22, 21, 16, 15, 14, 13, 36, 37, 116, 114, 102, 12, 66, 11, 10, 44, 118, 117, 62, 55, 90, 89, 85, 61, 111, 108, 112, 56, 110, 107, 109, 53, 91, 87, 43, 54, 105, 104, 78, 52, 75, 67, 77, 31, 86, 84, 74, 49, 69, 68, 45, 50, 51, 35, 34, 106, 33, 32, 0, 113, 119, 115, 88, 42, 29, 30, 2, 1, 3, 4, 5, 6, 7, 8, 9, 83, 123, 122, 121, 120, 124]
        # 去掉换行和空格，并转换为合法的列表形式
        llm_array = [-1] * len(number_list)
        for index,name in enumerate(number_list):
            llm_array[name] = float((len(number_list)-1-index)/(len(number_list)-1))

        cha = [float(llm_array[i+1])-float(llm_array[i]) for i in range(len(llm_array)-1)]
        cha = ["0"] + cha
        print("原始差",cha)

        for index,i in enumerate(cha):
            #if index+1 < len(cha) and abs(float(i))>0 and abs( float(i) + float(cha[index+1]) / float(i) ) < 0.33:
            #    cha[index] = 0
            #    cha[index+1] = float(i)+float(cha[index+1])
            if(float(i)>0.33):
                cha[index] = 0.33
            elif(float(i) < -0.33):
                cha[index] = -0.33

        print("处理后差",cha)

        weight = ["-1"] * len(cha)
        weight[0] = float(llm_array[0])
        for i in range(len(cha)-1):
            weight[i+1] = float(weight[i]) + float(cha[i+1])
            if weight[i+1] >= 1:
                weight[i+1] = 1
        
        
        print(weight)
        '''
        #发给llm  #这里函数内部输出了一次llm，gt
        plcc,srcc,plcc_idx,srcc_idx,weight=eval_llm(gtscore,info,frame_path,client,eval_path)  #除了weight以外都是数，weight是得分数组
        '''
        final_score=np.array(weight)
        idx_gt_scores=convert_to_index(gtscore)
        idx_final_scores=convert_to_index(final_score)

        #已经全部处理完毕，输出llmscore和gtscore
        print("LLM score:",final_score,"\nGT score:",gtscore)

        print("ALL PLCC")
        plcc, plcc_p_value = pearsonr(final_score, gtscore)
        srcc, srcc_p_value = spearmanr(final_score, gtscore)
        plcc_idx, plcc_idx_p_value = pearsonr(idx_final_scores, idx_gt_scores)
        srcc_idx, srcc_idx_p_value = spearmanr(idx_final_scores, idx_gt_scores)
        
        outlog = f"""
        (PLCC): {plcc}, p-value: {plcc_p_value}, Index PLCC: {plcc_idx}, p-value: {plcc_idx_p_value}
        (SRCC): {srcc}, p-value: {srcc_p_value}, Index SRCC: {srcc_idx}, p-value: {srcc_idx_p_value}
        """
        print(outlog)

        #llm评分的数组
        save_weight(llm_score_path,weight)
        # weight=load_weight(llm_score_path)*10000*30/ 20.9117524

        #重复输出了一次llm评分
        print("Weight:",weight)

        #绘图
        plot_score(gtscore, weight, 'GT', 'LLM', f'{video_name}_score:', os.path.join(result_path, "score.png"))

        # if index == 1: return
            
        #返回剩余的llm积分
        get_data(client=client)

if __name__ == "__main__":
    main()
