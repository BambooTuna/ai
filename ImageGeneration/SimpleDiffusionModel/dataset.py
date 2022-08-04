import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(source_root, width=64, height=64):
    data_transforms = [
        transforms.Resize((width, height)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # to [0, 1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # to [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)
    return torchvision.datasets.ImageFolder(source_root, data_transform)


def to_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
    ])
    return reverse_transforms(image)


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    image = reverse_transforms(image)
    plt.figure(figsize=[10, 4.2])
    plt.imshow(image)
    plt.show()
