import os
from datetime import datetime, timezone, timedelta

from torch.utils.tensorboard import SummaryWriter


def create_writer(log_root_path, dir_name=None):
    output_root = f'{log_root_path}/{datetime.now(timezone(timedelta(hours=+9), "JST")).strftime("%Y%m%d_%H%M%S")}'
    if dir_name is not None:
        output_root = f'{log_root_path}/{dir_name}'
    os.makedirs(output_root, exist_ok=True)
    writer = SummaryWriter(output_root)
    return writer

