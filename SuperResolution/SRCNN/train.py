import os
import json
import sys

# from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm

import torch
from torch import log10
from model import SRCNN
import torch.nn as nn
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

    "lr": 1e-4,
    "conv3_lr": 1e-5,
}


def train(config, dataloader: DataLoader, eval_dataloader: DataLoader, networks=None, device="cpu"):
    writer = create_writer(config["log_dir_path"], config["log_dir_name"])
    criterion = nn.MSELoss()

    if networks is None:
        (model, optimizer) = get_networks(config, device)
    else:
        (model, optimizer) = networks

    for epoch in tqdm(range(1, config["epochs"] + 1), "epochs"):
        loss, prediction, high_resolution, low_resolution = None, None, None, None
        model.train()
        for step, batch in enumerate(tqdm(dataloader, "steps")):
            (low_resolution, high_resolution) = batch
            low_resolution = low_resolution.to(device)
            high_resolution = high_resolution.to(device)

            optimizer.zero_grad()
            prediction = model(low_resolution)
            loss = criterion(prediction, high_resolution)
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

            writer.add_image("train/prediction", to_tensor_image(prediction), epoch)
            writer.add_image("train/high_resolution", to_tensor_image(high_resolution), epoch)
            writer.add_image("train/low_resolution", to_tensor_image(low_resolution), epoch)

            writer.add_scalar("loss", loss.item(), epoch)

        if epoch % config["eval_par_epoch"] == 0 and eval_dataloader is not None:
            evaluation(config, epoch, writer, eval_dataloader, criterion, model, device)


def get_networks(config, device="cpu"):
    checkpoint = config["checkpoint"]
    model = SRCNN().to(device)
    optimizer = Adam([{'params': model.conv1.parameters()},
                      {'params': model.conv2.parameters()},
                      {'params': model.conv3.parameters(), 'lr': config["conv3_lr"]}],
                     lr=config["lr"])

    if checkpoint is not None:
        model.load_state_dict(torch.load(f'{checkpoint}/model.cpt'))
        optimizer.load_state_dict(torch.load(f'{checkpoint}/optimizer.cpt'))
    return model, optimizer


def evaluation(config, epoch, writer, dataloader: DataLoader, criterion, model, device="cpu"):
    model.eval()
    val_loss, val_psnr = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            (low_resolution, high_resolution) = batch
            low_resolution = low_resolution.to(device)
            high_resolution = high_resolution.to(device)

            prediction = model(low_resolution)
            loss = criterion(prediction, high_resolution)
            val_loss += loss.data
            val_psnr += 10 * log10(1 / loss.data)
            if step + 1 >= config["eval_ave_counts"]:
                break
    writer.add_scalar("loss/ave", val_loss / len(dataloader), epoch)
    writer.add_scalar("psnr/ave", val_psnr / len(dataloader), epoch)
