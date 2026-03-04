from datasets import load_from_disk  # 注意这里导入的变化
from torchvision import transforms
from torch.utils.data import DataLoader

def get_loader(config):
    # 从本地路径加载，不再需要联网
    # config.data_dir 是cfg配置中指明的数据集本地保存路径。
    dataset = load_from_disk(config.data_dir)  # 加载本地保存的数据集

    preprocess = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)
    loader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    return loader