import argparse
import os
import json

# from tqdm import tqdm
import torchvision
from tqdm import tqdm_notebook as tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
from model import SimpleUnet, Loss, Generator, linear_beta_schedule, get_index_from_list
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import to_tensor_image, show_tensor_image
from logger import create_writer, timestamp


def main(config):
    print(config)


def train(config, dataloader: DataLoader, networks=None, device="cpu"):
    writer = create_writer(config["log_dir_path"], config["log_dir_name"])

    if networks is None:
        (model, optimizer, scaler, eval_img) = get_networks(config, device)
    else:
        (model, optimizer, scaler, eval_img) = networks

    criterion = Loss()
    for epoch in tqdm(range(1, config["epochs"] + 1), "epochs"):
        loss, prediction, high_resolution, low_resolution = None, None, None, None
        model.train()
        for step, batch in enumerate(tqdm(dataloader, "steps")):
            image = batch[0]
            optimizer.zero_grad()

            t = torch.randint(0, config["T"], (image.shape[0],), device=device).long()
            loss = criterion(model, image, t)

            loss.backward()
            # scaler.scale(loss).backward()
            optimizer.step()
            # scaler.step(optimizer)

            # scaler.update()

        if epoch % config["log_par_epoch"] == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()} ")
            writer.add_scalar("loss", loss.item(), epoch)

        if epoch % config["save_par_epoch"] == 0:
            base_path = f'{config["save_dir_path"]}/{str(epoch).zfill(6)}'
            os.makedirs(base_path, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "eval_img": eval_img,
            }, f'{base_path}.cpt')
            with open(f'{base_path}/config.json', 'w') as f:
                json.dump(config, f, ensure_ascii=False)

        if epoch % config["eval_par_epoch"] == 0:
            img = generate_image(model,
                                 eval_img,
                                 config["T"]
                                 )
            show_tensor_image(img)
            writer.add_image("eval/prediction", img, epoch)


def get_networks(config, device="cpu"):
    checkpoint = config["checkpoint"]
    model = SimpleUnet().to(device)
    optimizer = Adam(model.parameters(), lr=config["lr"])
    scaler = torch.cuda.amp.GradScaler()
    eval_img = torch.randn((1, 3, config["width"], config["height"]))

    if checkpoint is not None:
        d = torch.load(checkpoint)
        model.load_state_dict(d["model"])
        optimizer.load_state_dict(d["optimizer"])
        scaler.load_state_dict(d["scaler"])
        eval_img = d["eval_img"]
    return model, optimizer, scaler, eval_img.to(device)


@torch.no_grad()
def generate_image(model,
                   eval_img,
                   t_max):
    img = eval_img
    num_images = 10
    stepsize = int(t_max / num_images)

    g = Generator(t_max)
    images = []
    for t in range(0, t_max)[::-1]:
        img = g(model, img, t)
        if t % stepsize == 0:
            images.append(img.detach().cpu())
    return torchvision.utils.make_grid(to_tensor_image(torch.cat(images, dim=0)), nrow=len(images), padding=0)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    t = timestamp()

    parser = argparse.ArgumentParser(description="train")
    parser.add_argument('--save_dir', type=str, required=False, default=f"./output/{t}", help='save dir path')
    parser.add_argument('--save_dir', type=str, required=False, default=f"./output/{t}", help='save dir path')
    parser.add_argument('--checkpoint', type=str, required=False, default=None, help='checkpoint path')
    args = parser.parse_args()

    default_config = {
        "epochs": 1000,
        "log_dir_path": "./logs",
        "log_dir_name": t,

        "log_par_epoch": 1,

        "save_par_epoch": 5,
        "save_dir_path": args.save_dir,
        "checkpoint": args.checkpoint,

        "eval_par_epoch": 1,

        "width": 64,
        "height": 64,
        "batch_size": 128,
        "T": 300,
        "lr": 0.001,
    }

    main(default_config)
