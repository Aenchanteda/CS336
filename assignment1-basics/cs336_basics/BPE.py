import torch.nn as nn
import os
import subprocess
from pathlib import Path
import wget
import urllib
import tarfile
import gzip
import shutil
from cs336_basics import config

def download_and_extract_file(url, output_path, extracted_dir=None):
    """
    下载文件，如果需要则解压
    
    Args:
        url: 下载URL
        output_path: 保存路径
        extracted_dir: 解压目录（如果需要）
    
    Returns:
        最终文件路径
    """
    # 如果文件已存在，检查是否需要解压
    if os.path.exists(output_path):
        print(f"文件已存在: {output_path}")
    else:
        # 创建目录
        output_dir = os.path.dirname(output_path)
        if output_dir:  # 如果目录路径不为空
            os.makedirs(output_dir, exist_ok=True)
        try:
            print(f"正在从 {url} 下载...")
            wget.download(url, out=output_path)
            print(f"\n下载完成: {output_path}")
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"下载失败: {e}")
            return None
    
    # 检查是否需要解压
    file_path = Path(output_path)
    suffixes = file_path.suffixes
    
    # 处理 .gz 文件（gzip压缩）
    if '.gz' in suffixes:
        # 移除 .gz 后缀得到解压后的文件名
        decompressed_path = str(file_path).replace('.gz', '')
        if not os.path.exists(decompressed_path):
            try:
                print(f"开始解压 {output_path}...")
                with gzip.open(output_path, 'rb') as f_in:
                    with open(decompressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"解压完成: {decompressed_path}")
                return decompressed_path
            except (gzip.BadGzipFile, IOError) as e:
                print(f"解压失败: {e}")
                return output_path
        else:
            print(f"解压文件已存在: {decompressed_path}")
            return decompressed_path
    
    # 处理 tar 文件
    elif any(ext in suffixes for ext in ['.tar', '.bz2', '.xz']):
        if extracted_dir:
            os.makedirs(extracted_dir, exist_ok=True)
            try:
                print(f"开始解压 {output_path}...")
                with tarfile.open(output_path, 'r:*') as tar:
                    tar.extractall(path=extracted_dir)
                print(f"解压完成: {extracted_dir}")
                return extracted_dir
            except (tarfile.TarError, FileNotFoundError) as e:
                print(f"解压失败: {e}")
                return output_path
        else:
            return output_path
    
    # 无需解压
    else:
        print(f"无需解压，直接使用: {output_path}")
        return output_path

def download_dataset_tinystories():
    """
    下载TinyStories训练集和验证集，如果需要则解压
    
    Returns:
        tuple: (train_file_path, val_file_path) 两个文件的路径
    """
    print("开始下载TinyStories数据集...")
    
    # 创建目录
    os.makedirs(config.Data_RAW, exist_ok=True)
    os.makedirs(config.Data_Processed, exist_ok=True)
    
    # 定义文件路径
    train_file_path = os.path.join(config.Data_RAW, "TinyStoriesV2-GPT4-train.txt")
    val_file_path = os.path.join(config.Data_RAW, "TinyStoriesV2-GPT4-valid.txt")
    
    # 下载并解压训练集
    print("\n=== 处理训练集 ===")
    train_path = download_and_extract_file(
        config.URL,
        train_file_path
    )
    
    # 下载并解压验证集
    print("\n=== 处理验证集 ===")
    val_path = download_and_extract_file(
        config.URL_Tiny_Valid,
        val_file_path
    )
    
    if train_path and val_path:
        print(f"\n✓ 所有文件准备完成!")
        print(f"  训练集: {train_path}")
        print(f"  验证集: {val_path}")
        return train_path, val_path
    else:
        print("\n✗ 下载或解压过程中出现错误")
        return None, None

def input_file():
    input_path = os.path.join(config.Data_RAW, "TinyStories.txt")
    return input_path

if __name__ == "__main__":
    train_path, val_path = download_dataset_tinystories()
    if train_path and val_path:
        print(f"\n返回路径:")
        print(f"  Train: {train_path}")
        print(f"  Val: {val_path}")