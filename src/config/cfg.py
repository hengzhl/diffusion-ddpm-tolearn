from dataclasses import dataclass
import torch

@dataclass
class Config:
    image_size = 256
    image_channels = 3

    num_workers = 4
    train_batch_size = 16
    eval_batch_size = 8

    num_epochs = 150
    start_epoch = 0
    learning_rate = 1e-4

    logging_steps = 40
    
    diffusion_timesteps = 1000
    save_image_epochs = 5
    save_model_epochs = 10

    dataset = 'alkzar90/croupier-mtg-dataset'               # 数据集名称，如Hugging Face 上的 "alkzar90/croupier-mtg-dataset" 数据集
    data_dir = f'models/{dataset.split("/")[-1]}/data'      # 数据集本地保存路径
    output_dir = f'models/{dataset.split("/")[-1]}'
    resume = f"models/{dataset.split('/')[-1]}/unet256_e69.pth"              # N or Path of checkpoint to resume from
    checkpoint_path = f"models/{dataset.split('/')[-1]}/unet256_e129.pth"    # 模型权重文件路径，用于模型测试
    generate_dir = f"models/{dataset.split('/')[-1]}/generated_images"       # 模型测试生成结果的保存路径

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 0


cfg = Config()