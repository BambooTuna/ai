import os
from tqdm import tqdm

import torch
from model import SRCNN
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

default_config = {
    "epochs": 1000,
    "save_par_epoch": 5,
    "save_dir_path": "./output/20220101",
    "checkpoint": None,

    "lr": 1e-4,
    "conv3_lr": 1e-5,
}


def train(config, dataloader: DataLoader, device):
    criterion = nn.MSELoss()
    (model, optimizer) = get_networks(config, device)
    model.train()

    for epoch in tqdm(range(config.epochs), "epochs"):
        for step, batch in enumerate(tqdm(dataloader, "steps")):
            (low_resolution, high_resolution) = batch

            optimizer.zero_grad()
            prediction = model(low_resolution)
            loss = criterion(prediction, high_resolution)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % config.save_par_epoch == 0 and step == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()} ")
                os.makedirs(f"{config.save_dir_path}/{epoch}", exist_ok=True)
                torch.save(model.state_dict(), f"{config.save_dir_path}/{epoch}/model.cpt")
                torch.save(optimizer.state_dict(), f"{config.save_dir_path}/{epoch}/optimizer.cpt")


def get_networks(config, device):
    model = SRCNN().to(device)
    optimizer = Adam([{'params': model.conv1.parameters()},
                      {'params': model.conv2.parameters()},
                      {'params': model.conv3.parameters(), 'lr': config.conv3_lr}],
                     lr=config.lr)
    return model, optimizer
