from torchvision import transforms
import glob
from torch.utils.data import Dataset
from PIL import Image


def to_tensor_image(image):
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    return image


def pil_loader(path):
    return Image.open(path).convert("RGB")


class DatasetFolderPairs(Dataset):
    def __init__(self, source_root, scale=4, width=256, height=256, limit=None):
        super().__init__()
        self.filenames = glob.glob(f"{source_root}/**/*.jpg", recursive=True)
        if limit is not None:
            self.filenames = self.filenames[:limit]

        self.before_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # to [0, 1]
        ])
        self.low_resolution_transforms = transforms.Compose([
            transforms.Resize((width//scale, height//scale)),
            transforms.Resize((width, height))
        ])

    def __getitem__(self, index):
        img = pil_loader(self.filenames[index])
        img = self.before_transform(img)
        return self.low_resolution_transforms(img), img

    def __len__(self):
        return len(self.filenames)
