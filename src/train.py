import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from src.data.loader import get_loader
from src.model.unet import UNet
from src.scheduler.ddpm import DDPMPipeline
from src.utils.common import postprocess, create_images_grid
from src.config.cfg import cfg


def evaluate(config, epoch, pipeline, model,writer):
    # Perform reverse diffusion process with noisy images.
    noisy_sample = torch.randn(
        config.eval_batch_size,
        config.image_channels,
        config.image_size,
        config.image_size).to(config.device)

    # Reverse diffusion for T timesteps
    images = pipeline.sampling(model, noisy_sample, device=config.device)

    # Postprocess and save sampled images
    images = postprocess(images)
    image_grid = create_images_grid(images, rows=2, cols=3)

    grid_save_dir = Path(config.output_dir, "samples")
    grid_save_dir.mkdir(parents=True, exist_ok=True)
    image_grid.save(f"{grid_save_dir}/{epoch:04d}.png")

    from torchvision.transforms.functional import to_tensor
    writer.add_image(f"Generated_Images", to_tensor(image_grid), epoch)


def main():
    writer = SummaryWriter(log_dir=Path(cfg.output_dir, "logs"))

    train_loader = get_loader(cfg)

    model = UNet(image_size=cfg.image_size,
                 input_channels=cfg.image_channels).to(cfg.device)

    print("Model size: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                              T_max=len(train_loader) * cfg.num_epochs,
                                                              last_epoch=-1,
                                                              eta_min=1e-9)

    # Load checkpoint if resume is specified
    if cfg.resume:
        checkpoint = torch.load(cfg.resume, map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        cfg.start_epoch = checkpoint['epoch'] + 1
    #for param_group in optimizer.param_groups:
    #    param_group['lr'] = cfg.learning_rate

    diffusion_pipeline = DDPMPipeline(beta_start=1e-4, beta_end=1e-2, num_timesteps=cfg.diffusion_timesteps)

    global_step = cfg.start_epoch * len(train_loader)

    # Training loop
    for epoch in range(cfg.start_epoch, cfg.num_epochs):
        progress_bar = tqdm(total=len(train_loader))
        progress_bar.set_description(f"Epoch {epoch}")

        mean_loss = 0

        model.train()
        for step, batch in enumerate(train_loader):
            original_images = batch['images'].to(cfg.device)
            batch_size = original_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, diffusion_pipeline.num_timesteps, (batch_size,),
                                      device=cfg.device).long()

            # Apply forward diffusion process at the given timestep
            noisy_images, noise = diffusion_pipeline.forward_diffusion(original_images, timesteps)
            noisy_images = noisy_images.to(cfg.device)
            # Predict the noise residual
            noise_pred = model(noisy_images, timesteps)
            loss = F.mse_loss(noise_pred, noise)
            # Calculate new mean on the run without accumulating all the values
            mean_loss = mean_loss + (loss.detach().item() - mean_loss) / (step + 1) # 实时平均值计算技巧，节省内存
            loss.backward()         # 反向传播计算梯度

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪，防止梯度爆炸
            optimizer.step()        # 更新模型参数
            lr_scheduler.step()     # 更新学习率
            optimizer.zero_grad()   # 清空梯度

            if global_step % cfg.logging_steps == 0:
                writer.add_scalar("Loss/train", mean_loss, global_step) # 使用 mean_loss 更平滑
                writer.add_scalar("LR", lr_scheduler.get_last_lr()[0], global_step)

            progress_bar.update(1) 
            logs = {"loss": mean_loss, "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1

        # Evaluation
        if (epoch + 1) % cfg.save_image_epochs == 0 or epoch == cfg.num_epochs - 1:
            model.eval()
            evaluate(cfg, epoch, diffusion_pipeline, model, writer)

        if (epoch + 1) % cfg.save_model_epochs == 0 or epoch == cfg.num_epochs - 1:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'parameters': cfg,
                'epoch': epoch
            }
            torch.save(checkpoint, Path(cfg.output_dir,
                                        f"unet{cfg.image_size}_e{epoch}.pth"))

if __name__ == "__main__":
    main()