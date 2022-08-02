import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

from dataset import DatasetFolderPairs


def load_dataset(source_root, width=64, height=64):
    data_transforms = [
        transforms.Resize((width, height)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # to [0, 1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # to [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)
    return torchvision.datasets.ImageFolder(source_root, data_transform)


def load_resolution_pair_dataset(source_root, pairs=((64, 64), (256, 256)), width=256, height=256):
    before_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
    ])
    data_transforms = [
        transforms.Compose([transforms.Resize((w, h)), transforms.Resize((width, height))])
        for (w, h) in pairs
    ]
    after_transform = transforms.Compose([
        transforms.ToTensor(),  # to [0, 1]
        # transforms.Lambda(lambda t: (t * 2) - 1)  # to [-1, 1]
    ])
    return DatasetFolderPairs(source_root, before_transform, data_transforms, after_transform)


def show_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))


def to_tensor_image(image):
    reverse = transforms.Compose([
        # transforms.Lambda(lambda t: (t + 1) / 2),
    ])
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    image = reverse(image)
    return image


# You can create dataloader by under code
# from torch.utils.data import DataLoader
# dataset = load_dataset("/content/images")
# dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
