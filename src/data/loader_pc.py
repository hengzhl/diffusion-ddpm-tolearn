'''
数据集来自 Hugging Face 的 "alkzar90/croupier-mtg-dataset" 数据集，包含了 Magic: The Gathering 卡牌的图像和相关信息。
由于云端服务器无法连接外网,所以在PC本地下载数据集并保存到本地路径,手动传输至服务器,loader.py文件从本地路径加载数据集。
'''
from datasets import load_dataset
from src.config.cfg import cfg

dataset = load_dataset(cfg.dataset, split="train")  # 加载cfg.dataset数据集的训练部分
dataset.save_to_disk(cfg.data_dir)  # 保存到本地路径