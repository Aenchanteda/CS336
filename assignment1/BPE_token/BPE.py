import torch.nn as nn
import os
import subprocess
from pathlib import Path
import wget
import urllib
import tarfile
from . import config

def download_dataset_tinystories_valid():
    print("...尝试下载TinyStories_valid...")
    #如果存在则不进行后续操作
    if os.path.exists(config.Data_RAW_downloaded):
        print(f"数据集已存在：{config.Data_RAW_downloaded}")
        file_path = os.path.join(config.Data_RAW, "TinyStories.txt")
        return file_path
    #创建目录,下载数据路径
    os.makedirs(config.Data_RAW,exist_ok=True)
    os.makedirs(config.Data_Processed,exist_ok=True)

    file_path = os.path.join(config.Data_RAW,"TinyStories.txt")
    if not os.path.exists(file_path):#如果不存在则下载
        try:
            wget.download(config.URL_Tiny_Valid, out=file_path)
            print(f"从{config.URL_Tiny_Valid}下载...")
        except(urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"下载失败：{e}")
    else: 
        print("文件已存在，判断是否需要解压")  
    suffixes = Path(file_path).suffixes
    if any (a in suffixes for a in ['.tar','.gz','.bz2','.xz']):
        try:
            print("开始解压")
            with tarfile.open(file_path,'r') as tar:
                tar.extractall(path=config.Extracted_DATASET_PATH_2)
                print(f"数据集解压完成:{config.Extracted_DATASET_PATH_2}")    
        except (tarfile.TarError, FileNotFoundError) as e:
            print(f"解压失败{e}")
    else:
        print("无需解压，直接返回下载的文件的路径")

def input_file():
    input_path = os.path.join(config.Data_RAW, "TinyStories.txt")

    return input_path

if __name__ == "__main__":
    download_dataset_tinystories_valid()