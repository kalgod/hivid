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
import json
from moviepy.editor import VideoFileClip

bot_number = 0
total_frame = 0

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

tokens_lcy = {
    'p-b': "XEHnzZntyGNmUjM8kQK7oQ==", 
    'p-lat': "XOid9IBNkhyF0CraqLY8HfctbrM8gkaBJYg2Q4VKzA==",
    # 'formkey': 'c8ceeb7e8b2c67b13aceb888ddbfa20d',
}

class DataManager:
    def __init__(self):
        self.score_map = {}
        self.story_map = {}
        self.score_filename=""
        self.story_filename=""

    @staticmethod
    def get_numbers(paths):
        """
        从路径列表中提取数字。
        :param paths: 路径列表
        :return: 提取的数字列表
        """
        return [int(path.split('/')[-1].split('.')[0]) for path in paths]

    @staticmethod
    def get_key(numbers):
        """
        将输入的数字数组排序后转换为元组，用作字典的键。
        这样可以保证数组的顺序无关。
        """
        return ','.join(map(str, sorted(numbers)))

    def update_score(self, numbers, scores):
        """
        更新或添加分数到对应的数字数组。
        :param numbers: 数字数组，例如 [1, 2, 3]
        :param scores: 每个数字对应的分数数组，例如 [10, 20, 30]
        你将 self.score_map["1,2,3"] 赋值为 {3: 30, 1: 10, 2: 20} 
        """
        if len(numbers) != len(scores):
            raise ValueError("数字数组和分数数组长度必须相等！")
        key = self.get_key(numbers)
        self.score_map[key] = dict(zip(numbers, scores))

    def update_story(self, numbers, story_p, story_t, rating):
        """
        更新或添加故事到对应的数字数组。
        :param numbers: 数字数组，例如 [1, 2, 3]
        :param scores: 故事 "a"
        你将 self.score_map["1,2,3"] 赋值为 "a" 
        """
        key = self.get_key(numbers)
        self.story_map[key] = [story_p,story_t,rating]

    def get_score(self, numbers):
        """
        获取对应数字数组的分数。
        :param numbers: 数字数组，例如 [1, 2, 3]
        :return: 分数数组，例如 [10, 20, 30]，如果不存在则返回 None
        """
        key = self.get_key(numbers)
        if key in self.score_map:
            score_dict = self.score_map[key]
            return [score_dict[str(num)] for num in numbers]
        return None

    def get_story(self, numbers):
        """
        获取对应数字数组的故事。
        :param numbers: 数字数组，例如 [1, 2, 3]
        :return: 故事描述，例如"a"，如果不存在则返回 None
        """  
        key = self.get_key(numbers)
        if key in self.story_map:
            return self.story_map[key]
        return (None, None, None)

    def key_exists_score(self, numbers):
        """
        判断数字数组对应的键是否存在。
        :param numbers: 数字数组，例如 [1, 2, 3]
        :return: 如果存在返回 True，否则返回 False
        """
        key = self.get_key(numbers)
        return key in self.score_map
    
    def key_exists_story(self, numbers):
        """
        判断数字数组对应的键是否存在。
        :param numbers: 数字数组，例如 [1, 2, 3]
        :return: 如果存在返回 True，否则返回 False
        """
        key = self.get_key(numbers)
        return key in self.story_map

    def save_score_to_file(self):
        """
        将字典保存到文件中，使用 JSON 格式。
        :param filename: 保存的文件名
        """
        with open(self.score_filename, 'w') as f:
            json.dump(self.score_map, f)

    def save_story_to_file(self):
        """
        将字典保存到文件中，使用 JSON 格式。
        :param filename: 保存的文件名
        """
        with open(self.story_filename, 'w') as f:
            json.dump(self.story_map, f)

    def load_score_from_file(self):
        """
        从文件中加载字典，使用 JSON 格式。
        :param filename: 加载的文件名
        """
        with open(self.score_filename, 'r') as f:
            self.score_map = json.load(f)

    def load_story_from_file(self):
        """
        从文件中加载字典，使用 JSON 格式。
        :param filename: 加载的文件名
        """
        with open(self.story_filename, 'r') as f:
            self.story_map = json.load(f)

data_man=DataManager()

#得到剩余的llm积分
def get_data(client):
    # Get chat data of all bots (this will fetch all available threads)
    # print(client.get_chat_history("claude-3.5-sonnet")['data'])
    # Get chat data of a bot (this will fetch all available threads)
    # print(client.get_chat_history())
    data = client.get_settings()
    print(data["messagePointInfo"]["messagePointBalance"])
    # exit(0)
    # print(client.get_available_creation_models())
    # print(client.get_available_bots())

#eval_llm使用
def send_message(image_path, client,info):
    global total_frame

    # 以下是第一个图片的描述，给了图片的数量和视频的信息
    message = f"""
    I have uploaded {len(image_path)} frames, each representing a video chunk of 1 seconds. You first extract the frame number attached below the image content. The original video informations are {info[0]}
    This video tells a story about:
    {info[1]}
    I have divided this video into multiple consecutive segments,I divided this video into multiple consecutive segments, each lasting {len(image_path)} seconds. The plots of these segments are as follows:
    {info[2]}
    Your task is as below:
    1. Based on the video information background, first summarize some keywords that may attract viewers. Then based on your keywords, analyze each image content.
    2. Based on your analysis, on a scale of integer (0,100), rate all the {len(image_path)} frames such that higher number exhibits higher interestingness score. Different frames can yield the same scores. The score difference can be significant.
    Remember: There are {total_frame} frames in total.

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

    bot0_info=["gpt4_o_mini",649104377]
    bot1_info=["gpt4_o_mini",930753558]
    bot1_info=["gpt4_o_mini",649104377]

    # bot0_info=["gpt4_o_mini_128k",667358994]
    # bot1_info=["gpt4_o_mini_128k",667360175]

    bot0_info=["gpt4_o",989772183]
    bot1_info=["gpt4_o",989770600]
    
    global bot_number

    if bot_number == 1:
        bot=bot0_info[0];chatId=bot0_info[1]
        bot_number = 0
    elif bot_number == 0:
        bot=bot1_info[0];chatId=bot1_info[1]
        bot_number = 1
    else:
        raise bot_number_error

    while True:
        try:
            for chunk in client.send_message(bot=bot, message=message, file_path=image_path, chatId=chatId, timeout=30): pass  # 处理每个 chunk
            break  # 如果成功发送，退出循环
        except RuntimeError as e:
            print(f"Runtime Error: {e}, retrying...")  # 打印错误信息
            
            # 切换 bot 和 chatId
            if bot_number == 1:
                bot=bot0_info[0];chatId=bot0_info[1]
                bot_number = 0
            elif bot_number == 0:
                bot=bot1_info[0];chatId=bot1_info[1]
                bot_number = 1
            else:
                raise bot_number_error

        except Exception as e:
            print(f"An unexpected error occurred: {e}")  # 捕获其他异常
            # 切换 bot 和 chatId
            if bot_number == 1:
                bot=bot0_info[0];chatId=bot0_info[1]
                bot_number = 0
            elif bot_number == 0:
                bot=bot1_info[0];chatId=bot1_info[1]
                bot_number = 1
            else:
                raise bot_number_error

    res = chunk["text"] #从返回的chunk中取出了llm的返回值
    try:
        client.chat_break(bot=bot,chatId=chatId)
    except:
        print("chat_break error")
    sleep(5)
    return res

#mkstory使用
def send_message_story(image_path, client,info,story_last):

    get_data(client=client)
    global total_frame

    if story_last == None:
        message = f"""
    I have uploaded {len(image_path)} frames, each representing a video chunk of 1 seconds. They formed a continuous {len(image_path)} second video clip, and this vedio has {total_frame} frames in total.
    The original video informations are {info[0]}
    Your task is as below:
    1. Based on the frames and informations, analyze the content that this video clip attempts to convey. Then output your answer as "story_partial" and "story_total". (No more than 100 words, and those two should be the same.)
    2. Based on your analysis, predict the possibility of this video clip to be the most attracting clip throughout the entire video (on a scale of 0-100).After the json answer, explain your rating.
    
    Your answer must be a json format like this: 
    ```json
    [
        ("story_partial": "xxx"),
        ("story_total": "xxx"),
        ("rating": xxx)
    ]
    ```
    """
    else:
        message = f"""
    I have uploaded {len(image_path)} frames, each representing a video chunk of 1 seconds. They formed a continuous {len(image_path)} second video clip, and this vedio has {total_frame} frames in total. Before this vedio clip, this vedio tells a story about:
    {story_last}
    The original video informations are {info[0]}
    Your task is as below:
    1. Based on the frames and story and informations, analyze the content that this video clip attempts to convey.You only need to describe the facts, not analyze the meaning. Then output your answer as "story_partial". (No more than 100 words.)
    2. Based on the information you just analyzed and the content before this video clip, summarize what story this video has told so far (Do not separately explain the purpose of this paragraph).You only need to describe the facts, not analyze the meaning. Then output your answer as "story_total". (No more than {2*total_frame} words.)
    3. Based on your analysis, predict the possibility of this video clip to be the most attracting clip throughout the entire video (on a scale of 0-100).After the json answer, explain your rating.

    Your answer must be a json format like this: 
    ```json
    [
        ("story_partial": "xxx"),
        ("story_total": "xxx"),
        ("rating": xxx)
    ]
    ```
    """

    bot0_info=["gpt4_o_mini",649104377]
    bot1_info=["gpt4_o_mini",930753558]
    bot1_info=["gpt4_o_mini",649104377]

    # bot0_info=["gpt4_o_mini_128k",667358994]
    # bot1_info=["gpt4_o_mini_128k",667360175]

    bot0_info=["gpt4_o",989772183]
    bot1_info=["gpt4_o",989770600]
    
    global bot_number

    if bot_number == 1:
        bot=bot0_info[0];chatId=bot0_info[1]
        bot_number = 0
    elif bot_number == 0:
        bot=bot1_info[0];chatId=bot1_info[1]
        bot_number = 1
    else:
        raise bot_number_error

    while True:
        try:
            for chunk in client.send_message(bot=bot, message=message, file_path=image_path, chatId=chatId, timeout=30): pass  # 处理每个 chunk
            break  # 如果成功发送，退出循环
        except RuntimeError as e:
            print(f"Runtime Error: {e}, retrying...")  # 打印错误信息
            
            # 切换 bot 和 chatId
            if bot_number == 1:
                bot=bot0_info[0];chatId=bot0_info[1]
                bot_number = 0
            elif bot_number == 0:
                bot=bot1_info[0];chatId=bot1_info[1]
                bot_number = 1
            else:
                raise bot_number_error

        except Exception as e:
            print(f"An unexpected error occurred: {e}")  # 捕获其他异常
            # 切换 bot 和 chatId
            if bot_number == 1:
                bot=bot0_info[0];chatId=bot0_info[1]
                bot_number = 0
            elif bot_number == 0:
                bot=bot1_info[0];chatId=bot1_info[1]
                bot_number = 1
            else:
                raise bot_number_error

    res = chunk["text"] #从返回的chunk中取出了llm的返回值
    try:
        client.chat_break(bot=bot,chatId=chatId)
    except:
        print("chat_break error")
    sleep(5)
    return res

#给视频分块  暂时不看了
def extract_chunks(video_path, chunk_duration, frame_path, ref):
    # 从视频中提取帧，每个帧代表一个视频块
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    video = VideoFileClip(video_path)
    fps = np.round(video.fps)

    # 检查并清理帧文件夹
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
    else:
        os.system("rm -rf " + frame_path + "/*")

    # 动态计算需要处理的帧数
    print(f"{video_path} Video FPS: {fps}, total Frames in Video: {total_frames}, Ref Duration: {ref}, Ref Frames: {fps * ref}")

    if fps * (ref - 1) > total_frames:
        print("Changing FPS")
        fps = int(total_frames / ref)

    # 按帧逐步处理视频并直接保存帧
    saved_frame_count = 0
    for i, frame in enumerate(video.iter_frames()):
        if i % fps == 0:  # 每秒取一帧作为参考帧
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # 保存当前帧到目标路径
            modify_image(frame_bgr, f"frame: {saved_frame_count}", os.path.join(frame_path, f"{saved_frame_count}.png"))
            saved_frame_count += 1
        if saved_frame_count >= ref:  # 提取足够的帧后停止
            break

    print(f"Extracted {saved_frame_count} chunks from the video.")
    return saved_frame_count

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

#mkstory使用，解析LLM返回的JSON字符串                                                                                                                                             
def extract_story(res):
    # if ("json" not in res): return 0,None
    if ("[" not in res): return 0,None
    # if ("I'm unable to" in res or "unable to view" in res): return 0,None
    start_index = res.index('[')
    end_index = res.index(']') + 1
    # print(f"Start index: {start_index}, End index: {end_index}")

    # 提取 JSON 字符串
    json_str = res[start_index:end_index]
    # print(f"JSON string: {json_str}")

    # 查找 story_partial
    partial_start = json_str.find('"story_partial": "')  # 查找 variable 'story_partial'
    if partial_start != -1:  # 如果找到了
        partial_start += len('"story_partial": "')  # 移动到值的开始位置
        partial_end = json_str.find('"', partial_start)  # 找到值的结束位置
        story_partial = json_str[partial_start:partial_end]  # 提取值

    # 查找 story_total
    total_start = json_str.find('"story_total": "')  # 查找 variable 'story_total'
    if total_start != -1:  # 如果找到了
        total_start += len('"story_total": "')  # 移动到值的开始位置
        total_end = json_str.find('"', total_start)  # 找到值的结束位置
        story_total = json_str[total_start:total_end]  # 提取值

    # 查找 rating
    rating_start = json_str.find('"rating": ')  # 查找 variable 'story_total'
    if rating_start != -1:  # 如果找到了
        rating_start += len('"rating": ')  # 移动到值的开始位置
        rating_end = json_str.find('}', rating_start)  # 找到值的结束位置
        rating_part = json_str[rating_start:rating_end].split('}')[0].strip()
        rating = float(rating_part)

    return story_partial,story_total,rating

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

#存一个 NumPy 数组
def save_weight(path,weight):
    weight=np.array(weight)
    np.save(path,weight)

#用于加载存储在磁盘上的 NumPy 数组或其他数据（例如存储为 .npy 或 .npz 格式）的文件
def load_weight(path):
    weight=np.load(path)
    return weight

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

def modify_score(llm_array):
    sigma=5
    kernel_size=5
    k=kernel_size//2
    x=np.arange(-k,k+1)
    gaussian_kernel=np.exp(-0.5*(x/sigma)**2)
    gaussian_kernel/=gaussian_kernel.sum()
    llm_array = np.array([float(x) for x in llm_array])
    smoothed_array = np.convolve(llm_array, gaussian_kernel, mode='same')
    return smoothed_array.tolist()

def eval_llm_msort(gt_scores,info,frame_path,client,eval_path,eval_path_modified,m): #gt_scores是npy数组，info是文本，frame文件夹路径，client，eval.txt路径

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

    np_final_score=np.array(final_score)
    idx_gt_scores=convert_to_index(gt_scores)
    idx_final_scores=convert_to_index(np_final_score)

    #已经全部处理完毕，输出llmscore和gtscore
    print("LLM score:",np_final_score,"\nGT score:",gt_scores)

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

def eval_llm_msort_partial(gt_scores,info,frame_path,client,eval_path,eval_path_modified,m): #gt_scores是npy数组，info是文本，frame文件夹路径，client，eval.txt路径

    #以下建立了temp文件夹(路径为temp_path)，用于存储带有分数的frame
    frame_num=gt_scores.shape[0]*1  #理论上应该有的frames
    pre_frame_path=os.path.dirname(frame_path)  #指向该视频文件夹的frames文件夹的上一级
    path_list=os.listdir(frame_path)
    assert len(path_list)==frame_num  #assert用于判断两者是否相同，此处仅用于排除错误
    tmp_path=pre_frame_path+"/tmp"
    os.makedirs(tmp_path, exist_ok=True)    #temp_path 存储了带有分数的frame

    ### end chunk
    selected_num=5
    max_index = np.argmax(gt_scores)
    if (max_index<selected_num):
        start_time = 0
        end_time = selected_num*2
    elif (max_index>len(gt_scores)-selected_num):
        start_time = len(gt_scores)-selected_num*2
        end_time = len(gt_scores)
    else:
        start_time = max_index - selected_num
        end_time = max_index + selected_num

    gt_scores = gt_scores[start_time:end_time]

    path_list = []
    for i in range(start_time,end_time):
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

    number_list=np.array(number_list)
    number_list=number_list-np.min(number_list) #最小值为0
    print(number_list)

    final_score = [-1] * len(number_list)
    for index,name in enumerate(number_list):
        final_score[name] = float((len(number_list)-1-index)/(len(number_list)-1))

    np_final_score=np.array(final_score)
    idx_gt_scores=convert_to_index(gt_scores)
    idx_final_scores=convert_to_index(np_final_score)

    #已经全部处理完毕，输出llmscore和gtscore
    print("LLM score:",np_final_score,"\nGT score:",gt_scores)

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

def merge_sort(arr,m,client,info): #arr是[[],[],[]]的一堆groups
    # 基本情况：如果数组长度小于或等于1，则返回数组
    if len(arr) == 1:
        return merge(m,arr[0],client,info)  #这是一个[]

    # 找到数组的中间索引，整除
    mid = len(arr) // 2

    # 递归地对左右两半进行归并排序 5个的话分2+3,6分3+3
    left_half = merge_sort(arr[:mid],m,client,info)
    right_half = merge_sort(arr[mid:],m,client,info)

    # 合并已排序的左右两半
    return merge(m,left_half, right_half,client,info)

def merge(*args):  #承载了排序和合并两大功能    (m,1,client,info)或者(m,1,2,client,info).  注意这里过来的都是[1,2,3,4],返回也是[]
    path = os.path.dirname(args[1][0])
    merged = []
    if len(args) == 4:  #只有一个groups，直接排序输出
        m = args[0]
        client = args[2]
        info = args[3]
        if len(args[1]) <= m:
            eval_path = args[1]
            print("只有一项，而且个数不大于m")
            position = eval_llm_m(eval_path,client,info)
            for i in position:
                merged.append(os.path.join(path, f"{i}.png"))

    if len(args) == 5:  #两个groups
        m,remain1,remain2,client,info = args
        total = len(args[1]) + len(args[2])
        while len(merged) != total:
            if len(remain1)*len(remain2) == 0:
                merged = merged+remain1+remain2 #一项为零，其余直接继承
            elif len(remain1)+len(remain2) > m:   #如果不能一次性比较完
                eval_path = remain1[:m//2]+remain2[:m//2]
                print(len(remain1),len(remain2),len(eval_path),"正常两组项")
                position = eval_llm_m(eval_path,client,info)
                for i in range(m//2):
                    merged.append(os.path.join(path, f"{position[i]}.png"))
                    #merged.append(eval_path[position[i]])
                    if os.path.join(path, f"{position[i]}.png") in remain1:
                        remain1.remove(os.path.join(path, f"{position[i]}.png"))
                    else:
                        remain2.remove(os.path.join(path, f"{position[i]}.png"))
            else: #可以一次性比较完
                eval_path = remain1+remain2
                print(len(remain1),len(remain2),len(eval_path),"两组剩余项")
                position = eval_llm_m(eval_path,client,info)
                for i in range(len(eval_path)):
                    merged.append(os.path.join(path, f"{position[i]}.png"))
                    #merged.append(eval_path[position[i]])

    return merged

def eval_llm_m(image_path,client,info):
    global data_man
    name_set = set()
    for i in image_path:
        name_set.add(os.path.basename(i))

    last_res=""
    for j in range(5): #尝试五次llm请求
        frame=data_man.get_numbers(image_path)
        llm_score=data_man.get_score(frame)
        check=1
        res="Using Cached Results"
        if (llm_score==None):
            res = send_message(image_path=image_path, client=client,info=info) #image_path里面有4个图片的路径
            print(res)
            check,frame,llm_score=extract_score(res) #默认check返回1，frame返回了输出的帧的数字列表，score返回的是score数组
            frame = frame.tolist()
            llm_score = llm_score.tolist()
            data_man.update_score(frame,llm_score)
            data_man.save_score_to_file()
        else:
            print("Using cached llm score from data",frame,llm_score)

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

    # 将 frame 和 rating 组合成一个元组列表
    combined = list(zip(frame, llm_score))

    # 根据 rating（降序）和 frame（升序）进行排序
    sorted_combined = sorted(combined, key=lambda x: (-x[1], x[0]))

    # 提取排序后的 frame 名称
    sorted_frames = [frame for frame, rating in sorted_combined]

    print("排序",sorted_frames)

    return(sorted_frames)   #尽管gpt是乱序的，我们依然能找到……吧，直接返回图片的序号，重要性递减，eg[1,5,4,3,52,100]

def mkstory(info,frame_path,client,m):

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

    story_last=None
    story_list=[]
    for image_path in groups:
        global data_man
        last_res=""
        for j in range(5): #尝试五次llm请求
            frame=data_man.get_numbers(image_path)
            story_p,story_t,rating=data_man.get_story(frame)
            check=1
            res="Using Cached Results"
            if (story_p==None and story_t==None):
                res = send_message_story(image_path=image_path, client=client,info = info,story_last=story_last)
                story_p,story_t,rating=extract_story(res)
                print(res)
                data_man.update_story(frame,story_p,story_t,rating)
                data_man.save_story_to_file()
            else:
                print("Using cached llm story from data",frame,"\n",story_p,"\n",story_t,"\n",rating)

            if (check==1 and res!=last_res):
                print("合法",frame,"\n",story_p,"\n",story_t,"\n",rating)
                story_list.append(story_p)
                story_last = story_t
                break #合法的返回值,   #无重复元素
            else:
                print("No json or wrong answer, repeat!!!!!")
                #sleep(5)
            if (check==0): 
                print("Error")
                exit(0)
            last_res=res
            print("\n###################################################\n###################################################\n")
            #完成了这个组！
       
    return story_t,story_list

def send_message_prescore(client,story,story_list):
    #predict the possibility of every segment to be the most attracting clip throughout the entire video
    #rate all the segments such that higher number exhibits higher interestingness score
    get_data(client=client)

    message = f"""
    I have a video here that tells the following story:
    {story}
    I have divided this video into multiple consecutive segments. The plots of these segments are as follows:
    {story_list}
    Your task is as below:
    1. Based on your analysis, on a scale of integer (0,100), rate all the segments such that higher number exhibits higher interestingness score. Different segments can yield the same scores. The score difference can be significant, with at least one score of 0 and one score of 100
    2. After the json answer, explain your rating.
    Your answer must be a json format like this: 
    ```json
    [
        ("segment_1": xxx),
        ("segment_2": xxx),
        ("segment_3": xxx)
    ]
    ```
    """

    bot0_info=["gpt4_o_mini",649104377]
    bot1_info=["gpt4_o_mini",930753558]
    bot1_info=["gpt4_o_mini",649104377]

    # bot0_info=["gpt4_o_mini_128k",667358994]
    # bot1_info=["gpt4_o_mini_128k",667360175]

    bot0_info=["gpt4_o",989772183]
    bot1_info=["gpt4_o",989770600]
    
    global bot_number

    if bot_number == 1:
        bot=bot0_info[0];chatId=bot0_info[1]
        bot_number = 0
    elif bot_number == 0:
        bot=bot1_info[0];chatId=bot1_info[1]
        bot_number = 1
    else:
        raise bot_number_error

    while True:
        try:
            for chunk in client.send_message(bot=bot, message=message, chatId=chatId, timeout=30): pass  # 处理每个 chunk
            break  # 如果成功发送，退出循环
        except RuntimeError as e:
            print(f"Runtime Error: {e}, retrying...")  # 打印错误信息
            
            # 切换 bot 和 chatId
            if bot_number == 1:
                bot=bot0_info[0];chatId=bot0_info[1]
                bot_number = 0
            elif bot_number == 0:
                bot=bot1_info[0];chatId=bot1_info[1]
                bot_number = 1
            else:
                raise bot_number_error

        except Exception as e:
            print(f"An unexpected error occurred: {e}")  # 捕获其他异常
            # 切换 bot 和 chatId
            if bot_number == 1:
                bot=bot0_info[0];chatId=bot0_info[1]
                bot_number = 0
            elif bot_number == 0:
                bot=bot1_info[0];chatId=bot1_info[1]
                bot_number = 1
            else:
                raise bot_number_error

    res = chunk["text"] #从返回的chunk中取出了llm的返回值
    print(res)
    try:
        client.chat_break(bot=bot,chatId=chatId)
    except:
        print("chat_break error")

    if ("[" not in res): return 0,None
    start_index = res.index('[')
    end_index = res.index(']') + 1
    # 提取 JSON 字符串
    json_str = res[start_index:end_index]

    lines = json_str.splitlines()
    rating = []
    for line in lines:
        start = line.find('": ')
        if start == -1 :
            continue
        start += len('": ')
        end = line.find('}',start)
        rating.append(float(line[start:end].strip()))
        print(line,float(line[start:end].strip()))

    return rating

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
    client = PoeApi(tokens=tokens_lcy)  # Adjust this to use the correct tokens if necessary
    get_data(client=client)
    
    #这里是要分析的视频
    video_list = os.listdir("./result_msort")
    video_list = os.listdir("./dataset/youtube")
    # video_list=["video_115","video_1779","video_3526","video_6112","video_6843","video_7474","video_17477","video_28190","video_28478","video_3933","video_29810","video_11969"]
    # video_list = ["video_25090", "video_21572","video_7242","video_28903","video_30442", "video_17691", "video_5404", "video_29385", "video_5405","video_23163"] 
    video_list=["video_25090"]
    plcc_all = []
    plcc_modify_all = []
    selected_video = []

    for (index, video_name) in enumerate(video_list): #enumerate函数同时遍历索引和目标，这里就是直接遍历video_list的每一项，index从0开始
        print(f"Processing video {video_name} ({index + 1}/{len(video_list)})")
        video_path = os.path.join(video_folder, video_name, f"{video_name}.mp4")   #指向.mp4文件
        frame_path = os.path.join(video_folder, video_name, "frames")    #指向该视频文件夹的frames文件夹
        gtscore_path = os.path.join(video_folder, video_name, "gtscore.npy")     #指向视频文件夹下的GT.npy
        info_path = os.path.join(video_folder, video_name, f"{video_name}_info.txt")    #指向info.txt

        result_path = os.path.join('./result_story',video_name) #结果存在本目录下的result_new/video_name文件夹下
        if not os.path.exists(result_path): os.makedirs(result_path)
        llm_score_path = os.path.join(result_path, "llmscore.npy") #把llm打分的存储路径存成llmscore.npy
        llm_score_path_modified = os.path.join(result_path, "llmscore_modified.npy") #把llm打分的存储路径存成llmscore_modified.npy
        eval_path = os.path.join(result_path, "eval.txt") #eval.txt
        eval_path_modified = os.path.join(result_path, "eval_modified.txt") #eval_modified.txt
        
        gtscore = load_weight(gtscore_path)
        global total_frame
        total_frame = gtscore.shape[0]

        if not os.path.exists(frame_path) or len(os.listdir(frame_path)) == 0:  #如果frame文件夹不存在或者为空
            chunk_num = extract_chunks(video_path=video_path, chunk_duration=chunk_duration, frame_path=frame_path,ref=gtscore.shape[0])
            print(f"extract {chunk_num} frames of {video_name}")
        # continue

        #文本信息整合在info列表
        info = []
        with open(info_path, 'r') as f:
            info.append(f.read())

        m=10

        global data_man
        score_path=os.path.join(result_path, f"dict_m_{m}.json") #dict.txt
        data_man.score_filename=score_path
        story_path=os.path.join(result_path, f"story_m_{m}.json") #story.txt
        data_man.story_filename=story_path
        data_man.score_map={}
        data_man.story_map={}
        if os.path.exists(score_path): data_man.load_score_from_file()
        if os.path.exists(story_path): data_man.load_story_from_file()

        story,story_list = mkstory(info,frame_path,client,m)
        print("生成完成。\n",story)
        info.append(story)
        info.append(story_list)

        #prescore = send_message_prescore(client,story,story_list)
        #print("打分完成。\n",prescore)


        plcc,srcc,plcc_idx,srcc_idx,weight,weight_modified =  eval_llm_msort_partial(gtscore,info,frame_path,client,eval_path,eval_path_modified,m)  #除了weight以外都是数，weight是得分数组
        save_weight(llm_score_path,weight)
        data_man.save_score_to_file()

        weight=load_weight(llm_score_path)
        weight_modified=modify_score(weight)

        #'''
        #找到第一个极大值附近的frame索引
        selected_num=5
        max_index = np.argmax(gtscore) #最大值的索引
        if (max_index<selected_num):
            start_time = 0
            end_time = selected_num*2
        elif (max_index>len(gtscore)-selected_num):
            start_time = len(gtscore)-selected_num*2
            end_time = len(gtscore)
        else:
            start_time = max_index - selected_num
            end_time = max_index + selected_num
        gtscore = gtscore[start_time:end_time]
        #'''
        plcc_ori=pearsonr(weight,gtscore)[0]
        plcc_modify=pearsonr(weight_modified,gtscore)[0]
        # if (plcc_ori<0.1): continue
        selected_video.append(video_name)
        print(f"Video:{video_name} {info}, ori plcc:{plcc_ori}, modify plcc:{plcc_modify}\n")
        
        plcc_all.append(plcc_ori)
        plcc_modify_all.append(plcc_modify)

        #绘图
        plot_score(gtscore, weight, 'GT', 'LLM', f'{video_name}_score:', os.path.join(result_path, "score.png"))
        save_weight(llm_score_path_modified,weight_modified)
        plot_score(gtscore, weight_modified, 'GT', 'LLM', f'{video_name}_score:', os.path.join(result_path, "score_modified.png"))
        get_data(client=client)
    plcc_all = np.array(plcc_all)
    plcc_modify_all = np.array(plcc_modify_all)
    print("Selected Video",selected_video)
    print("Total Video",len(plcc_all),"Average PLCC:",np.mean(plcc_all),"Average PLCC Modified:",np.mean(plcc_modify_all))

if __name__ == "__main__":
    main()
