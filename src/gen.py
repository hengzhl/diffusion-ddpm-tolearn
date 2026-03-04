import torch
from pathlib import Path
from PIL import Image
from torchvision.utils import save_image, make_grid

from src.model.unet import UNet
from src.scheduler.ddpm import DDPMPipeline
from src.config.cfg import cfg

def postprocess(x):
    """
    将 [-1, 1] 的 Tensor 转换为 [0, 1]
    """
    x = (x + 1) / 2
    return torch.clamp(x, 0, 1)

def main():
    # 1. 初始化模型并加载权重
    print(f"Loading model from {cfg.checkpoint_path}...")
    model = UNet(
        image_size=cfg.image_size,
        input_channels=cfg.image_channels
    ).to(cfg.device)

    checkpoint = torch.load(
        cfg.checkpoint_path,
        map_location=cfg.device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # 2. 初始化 Pipeline
    pipeline = DDPMPipeline(num_timesteps=cfg.diffusion_timesteps)
    
    # 3. 准备初始噪声
    initial_noise = torch.randn(
        cfg.eval_batch_size,
        cfg.image_channels,
        cfg.image_size,
        cfg.image_size
    ).to(cfg.device)

    print("Start Sampling (this may take a while)...")
    
    # 4. 调用原生 sampling 方法并开启 save_all_steps
    with torch.no_grad():
        all_steps_images = pipeline.sampling(
            model, 
            initial_noise, 
            device=cfg.device, 
            save_all_steps=True
        )

    # 5. 处理帧数据
    frames = []
    print("Processing frames for GIF...")
    
    for i, x_t in enumerate(all_steps_images):
        if i % 20 == 0 or i == len(all_steps_images) - 1:
            grid = make_grid(postprocess(x_t), nrow=4)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            frames.append(Image.fromarray(ndarr))

    # --- 核心修改：设置每一帧的持续时间 ---
    # 普通帧间隔 100ms，最后一帧停留 2000ms (2秒)
    # 你可以根据需要调整这个数值
    durations = [100] * (len(frames) - 1) + [2000] 

    # 6. 执行保存操作
    output_dir = Path(cfg.generate_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存最后一帧
    final_image = postprocess(all_steps_images[-1])
    save_image(final_image, output_dir / "generated_final.png", nrow=4)

    # 保存 GIF
    gif_path = output_dir / "denoising_process.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations, # 传入列表实现差异化时长
        loop=0
    )

    print(f"\nSuccess!")
    print(f"Final image: {output_dir / 'generated_final.png'}")
    print(f"Animation GIF (with 2s pause): {gif_path}")

if __name__ == "__main__":
    main()