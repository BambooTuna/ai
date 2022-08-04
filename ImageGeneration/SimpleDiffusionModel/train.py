import os
import json

# from tqdm import tqdm
import torchvision
from tqdm import tqdm_notebook as tqdm

import torch
import torch.nn.functional as F
from torch import log10
from model import SimpleUnet, get_loss, linear_beta_schedule, get_index_from_list
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import to_tensor_image
from logger import create_writer

default_config = {
    "epochs": 1000,
    "log_dir_path": "./logs",
    "log_dir_name": "20220802",

    "save_par_epoch": 5,
    "save_dir_path": "./output/20220101",
    "checkpoint": None,

    "eval_par_epoch": 5,
    "eval_ave_counts": 100,

    "width": 64,
    "height": 64,
    "T": 300,
    "lr": 0.001,
}


def train(config, dataloader: DataLoader, eval_dataloader: DataLoader, networks=None, device="cpu"):
    writer = create_writer(config["log_dir_path"], config["log_dir_name"])

    if networks is None:
        (model, optimizer) = get_networks(config, device)
    else:
        (model, optimizer) = networks

    betas = linear_beta_schedule(timesteps=config["T"])
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    for epoch in tqdm(range(1, config["epochs"] + 1), "epochs"):
        loss, prediction, high_resolution, low_resolution = None, None, None, None
        model.train()
        for step, batch in enumerate(tqdm(dataloader, "steps")):
            (image) = batch

            optimizer.zero_grad()

            t = torch.randint(0, config["T"], (image.shape[0],), device=device).long()
            loss = get_loss(model, image, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device)

            loss.backward()
            optimizer.step()

        if epoch % config["save_par_epoch"] == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()} ")
            base_path = f'{config["save_dir_path"]}/{str(epoch).zfill(6)}'
            os.makedirs(base_path, exist_ok=True)
            torch.save(model.state_dict(), f'{base_path}/model.cpt')
            torch.save(optimizer.state_dict(), f'{base_path}/optimizer.cpt')
            with open(f'{base_path}/config.json', 'w') as f:
                json.dump(config, f, ensure_ascii=False)

            img = generate_image(model,
                                 config["T"], config["width"], config["height"],
                                 betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance,
                                 device
                                 )

            writer.add_image("train/prediction", to_tensor_image(img), epoch)
            writer.add_scalar("loss", loss.item(), epoch)

        if epoch % config["eval_par_epoch"] == 0 and eval_dataloader is not None:
            evaluation(config, epoch, writer, eval_dataloader, model, device)


def get_networks(config, device="cpu"):
    checkpoint = config["checkpoint"]
    model = SimpleUnet().to(device)
    optimizer = Adam(model.parameters(), lr=config["lr"])

    if checkpoint is not None:
        model.load_state_dict(torch.load(f'{checkpoint}/model.cpt'))
        optimizer.load_state_dict(torch.load(f'{checkpoint}/optimizer.cpt'))
    return model, optimizer


def evaluation(config, epoch, writer, dataloader: DataLoader, model, device="cpu"):
    model.eval()
    val_loss, val_psnr = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            (image) = batch
            t = torch.randint(0, config["T"], (image.shape[0],), device=device).long()
            loss = get_loss(model, image, t, device)
            val_loss += loss.data
            val_psnr += 10 * log10(1 / loss.data)
            if step + 1 >= config["eval_ave_counts"]:
                break
    writer.add_scalar("loss/ave", val_loss / len(dataloader), epoch)
    writer.add_scalar("psnr/ave", val_psnr / len(dataloader), epoch)


@torch.no_grad()
def sample_timestep(model, x, t, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance):
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def generate_image(model,
                   t_max, width, height,
                   betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance,
                   device):
    img = torch.randn((1, 3, width, height), device=device)
    num_images = 10
    stepsize = int(t_max / num_images)

    images = []
    for i in range(0, t_max)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(model, img, t, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas,
                              posterior_variance)
        if i % stepsize == 0:
            images.append(img.detach().cpu())
    fakes_random = torchvision.utils.make_grid(torch.cat(images, dim=1), nrow=len(images), padding=0)
