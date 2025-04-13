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

bot_number = 0

# Tokens for Poe API (update with actual tokens if needed)
tokens = {
    'p-b': "i7JZq6B4feBIrYx2Ao8ewQ==", 
    'p-lat': "cuHNeEA7iTTHDGTzlgzdCa1b6niVntYU6vSQmOqr5g==",
    'formkey': 'c045ce1e4e5ba81ffa20da079984c515',
}

tokens_jlc = {
    'p-b': "i7JZq6B4feBIrYx2Ao8ewQ==", 
    'p-lat': "47gSJb5WzzGjjkNEn6KkpzdZ0jFX6tFaiYFR1FzYHQ==",
    # 'formkey': 'c8ceeb7e8b2c67b13aceb888ddbfa20d',
}

#得到剩余的llm积分
def get_data(client):
    # Get chat data of all bots (this will fetch all available threads)
    # print(client.get_chat_history("claude-3.5-sonnet")['data'])
    # Get chat data of a bot (this will fetch all available threads)
    # print(client.get_chat_history())
    data = client.get_settings()
    print(data)
    # print(data["messagePointInfo"]["messagePointBalance"])
    # print(client.get_available_creation_models())
    # print(client.get_available_bots())

#eval_llm使用 
def send_message(image_path, client,info):
    # 以下是第一个图片的描述，给了图片的数量和视频的信息
    fixed_msg_1 = f"""
    I have uploaded {len(image_path)} frames, each representing a video chunk of 1 seconds. You first extract the frame number attached below the image content. The original video informations are{info}
    Your task is as below:
    1. Based on the video information background, first summarize some keywords that may attract viewers. Then based on your keywords, analyze each image content.
    2. Based on your analysis, on a scale of integer (1,100), rate all the {len(image_path)} frames that higher number means higher interestingness score. The scores can be continuous. Different frames can yield the same score.
    Your answer must be a json format like this: 
    ```json
    [
        ("frame": xxx, "rating": xxx),
        ("frame": xxx, "rating": xxx),
        ("frame": xxx, "rating": xxx)
    ]
    ```
    show with frame number ascending. Below your json answer, analyze each image content and exaplain your rating.
    """
    # 从第二个图片开始，就用这个模板（有参考分数）
    fixed_msg_2 = f"""
    I have uploaded {len(image_path)} frames, each representing a video chunk of 1 seconds. You first extract the frame number attached below the image content. The video informations are:{info}
    Your task is as below:
    1. Based on the video information background, first summarize some keywords that may attract viewers. Then based on your keywords, analyze each image content.
    2. Based on your analysis, on a scale of integer (1,100), rate all the {len(image_path)} frames that higher number means higher interestingness score. Note that some of the image already contains a rating below, such rating is fixed, you cannot change such rating. 
    These fixed rating represent the interestingness score regarding their content. Therefore the rest of the relative rating should refer to these fixed rating. The scores can be continuous. Different frames can yield the same score.
    Your answer must be a json format like this: 
    ```json
    [
        ("frame": xxx, "rating": xxx),
        ("frame": xxx, "rating": xxx),
        ("frame": xxx, "rating": xxx)
    ]
    ```
    show with frame number ascending. Below your json answer, analyze each image content and exaplain your rating.
    """
    if ("/0.png" in image_path[0]): message = fixed_msg_1
    else: message = fixed_msg_2
    frame = ["temp2.png"]
    if image_path is not None:
        frame = image_path

    # bot = "gpt4_o";chatId= 662480034
    # bot = "gpt4_o_128k";chatId= 649712483
    # bot = "gpt4_o_128k";chatId= 808082159
    bot = "gpt4_o_mini";chatId= 649104377
    # bot="gpt4_o_mini_128k";chatId=667358994
    # bot="claude-3.5-sonnet";chatId=685220083
    # bot="claude_3_opus";chatId=None
    # bot = "gpt4_o_128k";chatId= None
    sleep(5)
    
    global bot_number
    '''
    if bot_number == 1:
        bot="gpt4_o_mini_128k";chatId=667358994
        bot_number = 0
    elif bot_number == 0:
        bot="gpt4_o_mini_128k";chatId=667360175
        bot_number = 1
    else:
        raise bot_number_error
    '''
    '''
    if bot_number == 1:
        bot = "gpt4_o_128k";chatId= 649712483
        bot_number = 0
    elif bot_number == 0:
        bot = "gpt4_o_128k";chatId= 808082159
        bot_number = 1
    else:
        raise bot_number_error
    '''

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
    return 1,rating

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

    '''
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
    '''
    max_score = max(score)
    max_index = score.index(max_score)
    # 组合结果并排序
    final_indices = [max_index]

    return final_indices

def modify_score(llm_array):

    cha = [float(llm_array[i+1])-float(llm_array[i]) for i in range(len(llm_array)-1)]
    cha = ["0"] + cha
    print("原始差",cha)

    for index,i in enumerate(cha):
        #if index+1 < len(cha) and abs(float(i))>0 and abs( float(i) + float(cha[index+1]) / float(i) ) < 0.33:
        #    cha[index] = 0
        #    cha[index+1] = float(i)+float(cha[index+1])
        if(float(i)>33):
            cha[index] = 33
        elif(float(i) < -33):
            cha[index] = -33

    print("处理后差",cha)

    weight = ["-1"] * len(cha)
    weight[0] = float(llm_array[0])
    for i in range(len(cha)-1):
        weight[i+1] = float(weight[i]) + float(cha[i+1])
        if weight[i+1] >= 1:
            weight[i+1] = 1
        if weight[i+1] <= 0:
            weight[i+1] = 0
    
    return(weight)


def eval_llm(gt_scores,info,frame_path,client,eval_path,eval_path_modified): #gt_scores是npy数组，info是文本，frame文件夹路径，client，eval.txt路径
    frame_num=gt_scores.shape[0]*1
    pre_frame_path=os.path.dirname(frame_path)
    path_list=os.listdir(frame_path)
    assert len(path_list)==frame_num
    tmp_path=pre_frame_path+"/tmp"
    os.makedirs(tmp_path, exist_ok=True)
    image_path=[]
    eval_frame_num=1  #4个一组
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
                #sleep(5)
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
        #sleep(5)
    final_score=average_every_n_elements(final_score,1)



    np_final_score=np.array(final_score)
    idx_gt_scores=convert_to_index(gt_scores)
    idx_final_scores=convert_to_index(np_final_score)

    #已经全部处理完毕，输出llmscore和gtscore
    print("LLM score:",np_final_score,"\nGT score:",gt_scores)

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
    plcc, plcc_p_value = pearsonr(np_final_score, gt_scores)
    srcc, srcc_p_value = spearmanr(np_final_score, gt_scores)
    plcc_idx, plcc_idx_p_value = pearsonr(idx_final_scores, idx_gt_scores)
    srcc_idx, srcc_idx_p_value = spearmanr(idx_final_scores, idx_gt_scores)
    
    outlog = f"""
    (PLCC): {plcc}, p-value: {plcc_p_value}, Index PLCC: {plcc_idx}, p-value: {plcc_idx_p_value}
    (SRCC): {srcc}, p-value: {srcc_p_value}, Index SRCC: {srcc_idx}, p-value: {srcc_idx_p_value}
    """
    print(outlog)

    with open(eval_path, "w") as f:
        f.write(outlog)



    ###modified
    print("下面的modified")
    
    final_score_modified = modify_score(final_score) 

    np_final_score_modified=np.array(final_score_modified)
    idx_gt_scores=convert_to_index(gt_scores)
    idx_final_scores=convert_to_index(np_final_score)

    print("LLM score:",np_final_score_modified,"\nGT score:",gt_scores)
    print("ALL PLCC")
    plcc, plcc_p_value = pearsonr(np_final_score_modified, gt_scores)
    srcc, srcc_p_value = spearmanr(np_final_score_modified, gt_scores)
    plcc_idx, plcc_idx_p_value = pearsonr(idx_final_scores, idx_gt_scores)
    srcc_idx, srcc_idx_p_value = spearmanr(idx_final_scores, idx_gt_scores)
    
    outlog = f"""
    (PLCC): {plcc}, p-value: {plcc_p_value}, Index PLCC: {plcc_idx}, p-value: {plcc_idx_p_value}
    (SRCC): {srcc}, p-value: {srcc_p_value}, Index SRCC: {srcc_idx}, p-value: {srcc_idx_p_value}
    """
    print(outlog)
    with open(eval_path_modified, "w") as f:
        f.write(outlog)

    return plcc,srcc,plcc_idx,srcc_idx,np_final_score,np_final_score_modified

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
    score2_norm = np.array(score2) / 100
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
    video_list = ["video_11968","video_3526","video_8530","video_6112","video_26318","video_7474"]
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

        result_path = os.path.join('./result_window',video_name) #结果存在本目录下的result_window/video_name文件夹下
        if not os.path.exists(result_path): #没有就创造directory
            os.makedirs(result_path)
        llm_score_path = os.path.join(result_path, "llmscore.npy") #把llm打分的存储路径存成llmscore.npy
        llm_score_path_modified = os.path.join(result_path, "llmscore_modified.npy") #把llm打分的存储路径存成llmscore_modified.npy
        eval_path = os.path.join(result_path, "eval.txt") #eval.txt
        eval_path_modified = os.path.join(result_path, "eval_modified.txt") #eval_modified.txt

        #创造gt数组，然后输出gt的个数
        gtscore = load_weight(gtscore_path)
        print("num of gtscore:",gtscore.shape[0])

        # 第一次运行时已提取，故不再重复进行
        # chunk_num = extract_chunks(video_path=video_path, chunk_duration=chunk_duration, frame_path=frame_path)
        # print(f"extract {chunk_num} frames of {video_name}")

        #输出正在分析的视频的信息
        with open(info_path, 'r') as f:
            info = f.read()
        print(f"Processing {video_name}\n{info}")
        
        #发给llm  #这里函数内部输出了一次llm，gt
        plcc,srcc,plcc_idx,srcc_idx,weight,weight_modified=eval_llm(gtscore,info,frame_path,client,eval_path,eval_path_modified)  #除了weight以外都是数，weight是得分数组
        
        #llm评分的数组
        save_weight(llm_score_path,weight)
        # weight=load_weight(llm_score_path)*10000*30/ 20.9117524

        #重复输出了一次llm评分
        print("Weight:",weight)

        #绘图
        plot_score(gtscore, weight, 'GT', 'LLM', f'{video_name}_score:', os.path.join(result_path, "score.png"))

        #从这里再绘制一次后处理的图片，和PLCC
        print("从这里再绘制一次后处理的图片，和PLCC")

        #llm评分的数组
        save_weight(llm_score_path_modified,weight_modified)
        plot_score(gtscore, weight_modified, 'GT', 'LLM', f'{video_name}_score:', os.path.join(result_path, "score_modified.png"))

        # if index == 1: return
    
    #返回剩余的llm积分
    get_data(client=client)

if __name__ == "__main__":
    main()
