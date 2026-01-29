import os

URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
URL_Tiny_Valid = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"

DATA_Dir = "data"
Tiny_DATASET_NAME_valid = "TinyStories.txt"
OWT_DATASET_NAME = "OWT"

Data_RAW = os.path.join(DATA_Dir,"Raw")
Data_Processed = os.path.join(DATA_Dir,"Processed")

Data_RAW_downloaded = os.path.join(Data_RAW)

Extracted_DATASET_PATH_1 = os.path.join(DATA_Dir,"extracted",OWT_DATASET_NAME)
Extracted_DATASET_PATH_2 = os.path.join(DATA_Dir,"extracted",Tiny_DATASET_NAME_valid)



'''
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..



'''