#本程序生成256个随机数列表，并试图以从高到低数字排序

import os
import random
cnt=0

def eval_llm_msort(m,frame_num):
    client = " "
    frame_path = "./test/"
    info = " "

    path_list = []
    # 创建一个包含0到255的列表
    numbers = list(range(frame_num))
    # 随机打乱列表
    random.shuffle(numbers)
    print(numbers)

    for i in numbers:
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
    print(groups)

    ### end chunk
    #把所有的帧分成m个一组

    result = merge_sort(groups,m,client,info)     #返回了一个数组，[path1,path2^^^^^^]按重要性从高到低排列

    rank_list = []
    for i in range(len(result)):
        #把frame里面的图片加上rating，放到tmp文件夹
        cur_path = result[i]
        rank_list.append(os.path.basename(cur_path))

    #rank_list存储的全都是照片的名字（'0.png'），转换成数字
    number_list = [int(filename.split('.')[0]) for filename in rank_list]

    print(number_list)
    return 0

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
    global cnt
    cnt+=1
    print(image_path)

    sorted_numbers = sorted([int(file.split('/')[-1].split('.')[0]) for file in image_path], reverse=True)
    print(sorted_numbers)

    return(sorted_numbers)   #尽管gpt是乱序的，我们依然能找到……吧，直接返回图片的序号，重要性递减，eg[1,5,4,3,52,100]
    

def main():
    video_list = os.listdir("./dataset/youtube")
    # video_list=["video_115","video_1779","video_3526","video_6112","video_6843","video_7474","video_17477","video_28190","video_28478","video_3933","video_29810","video_11969"]
    # video_list = ["video_25090", "video_21572","video_7242","video_28903","video_30442", "video_17691", "video_5404", "video_29385", "video_5405","video_23163"] 
    # video_list=["video_28155","video_22500", "video_13514"]
    # video_list=["video_115"]
    plcc_all = []
    plcc_modify_all = []
    selected_video = []

    for (index, video_name) in enumerate(video_list): #enumerate函数同时遍历索引和目标，这里就是直接遍历video_list的每一项，index从0开始
        print(f"Processing video {video_name} ({index + 1}/{len(video_list)})")
        video_path = os.path.join(video_folder, video_name, f"{video_name}.mp4")   #指向.mp4文件
        frame_path = os.path.join(video_folder, video_name, "frames")    #指向该视频文件夹的frames文件夹
        gtscore_path = os.path.join(video_folder, video_name, "gtscore.npy")     #指向视频文件夹下的GT.npy
        info_path = os.path.join(video_folder, video_name, f"{video_name}_info.txt")    #指向info.txt

        result_path = os.path.join('./result_msort',video_name) #结果存在本目录下的result_new/video_name文件夹下
        if not os.path.exists(result_path): os.makedirs(result_path)
        llm_score_path = os.path.join(result_path, "llmscore.npy") #把llm打分的存储路径存成llmscore.npy
        llm_score_path_modified = os.path.join(result_path, "llmscore_modified.npy") #把llm打分的存储路径存成llmscore_modified.npy
        eval_path = os.path.join(result_path, "eval.txt") #eval.txt
        eval_path_modified = os.path.join(result_path, "eval_modified.txt") #eval_modified.txt
        
        gtscore = load_weight(gtscore_path)
        global total_frame
        total_frame = gtscore.shape[0]
        
        
        m = 8
        frame_num=100
        eval_llm_msort(m,frame_num)
        print(cnt)



if __name__ == "__main__":
    main()
