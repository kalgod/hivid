#可以打印gtscore.npy

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


def load_weight(path):
    weight=np.load(path)
    return weight

def main():
    #以下是一些初始变量，从命令行提取，查看help可以用 python qoe.py --help
    #不使用初始值时，用 python qoe.py -dataset xxxx(./dataset/下面的文件夹名) -chunk xxxx(秒数)
    parser = argparse.ArgumentParser(description="Extract frames from videos and process them.")  
    parser.add_argument("-dataset",type=str, default="youtube", help="Folder containing videos to process.")
    parser.add_argument("-chunk", type=float,default=1, help="Duration of each chunk in seconds.")
    args = parser.parse_args()

    video_folder = "./dataset/"+args.dataset
    video_name = "video_5"

    gtscore_path = os.path.join(video_folder, video_name, "gtscore.npy")     #指向视频文件夹下的GT.npy

    #打印gt
    gtscore = load_weight(gtscore_path)
    print(gtscore)

main()