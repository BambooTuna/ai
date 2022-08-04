import torchvision
from torchvision import transforms


def to_tensor_image(image):
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    return image


def load_dataset(source_root, width=64, height=64):
    data_transforms = [
        transforms.Resize((width, height)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # to [0, 1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # to [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)
    return torchvision.datasets.ImageFolder(source_root, data_transform)

