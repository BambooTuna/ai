import os
import json

# from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm

import torch
from torch import log10
from model import SRCNN
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils.logger import create_writer

default_config = {
    "epochs": 1000,
    "log_dir_path": "./log",

    "save_par_epoch": 5,
    "save_dir_path": "./output/20220101",
    "checkpoint": None,

    "eval_par_epoch": 5,

    "lr": 1e-4,
    "conv3_lr": 1e-5,
}


def train(config, dataloader: DataLoader, eval_dataloader: DataLoader, device):
    writer = create_writer(config["log_dir_path"])
    criterion = nn.MSELoss()
    (model, optimizer) = get_networks(config, device)

    for epoch in tqdm(range(1, config["epochs"] + 1), "epochs"):
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

            if epoch % config["save_par_epoch"] == 0 and step == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()} ")
                os.makedirs(f'{config["save_dir_path"]}/{epoch}', exist_ok=True)
                torch.save(model.state_dict(), f'{config["save_dir_path"]}/{epoch}/model.cpt')
                torch.save(optimizer.state_dict(), f'{config["save_dir_path"]}/{epoch}/optimizer.cpt')
                with open(f'{config["save_dir_path"]}/{epoch}/config.json', 'w') as f:
                    json.dump(config, f, ensure_ascii=False)

                writer.add_image("train/prediction", prediction, epoch)
                writer.add_image("train/high_resolution", high_resolution, epoch)
                writer.add_image("train/low_resolution", low_resolution, epoch)

                writer.add_scalar("loss", loss.item(), epoch)

            if epoch % config["eval_par_epoch"] == 0 and step == 0 and eval_dataloader is not None:
                evaluation(config, epoch, writer, eval_dataloader, criterion, model, device)


def get_networks(config, device):
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


def evaluation(config, epoch, writer, dataloader: DataLoader, criterion, model, device):
    model.eval()
    val_loss, val_psnr = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            (low_resolution, high_resolution) = batch
            low_resolution = low_resolution.to(device)
            high_resolution = high_resolution.to(device)

            prediction = model(low_resolution)
            loss = criterion(prediction, high_resolution)
            val_loss += loss.data
            val_psnr += 10 * log10(1 / loss.data)
    writer.add_scalar("loss/ave", val_loss / len(dataloader), epoch)
    writer.add_scalar("psnr/ave", val_psnr / len(dataloader), epoch)
    model.train()
