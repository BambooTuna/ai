import glob
from torch.utils.data import Dataset
from PIL import Image


def pil_loader(path):
    return Image.open(path).convert("RGB")


class DatasetFolderPairs(Dataset):
    def __init__(self, image_dir, before_transform, data_transforms, after_transform):
        super().__init__()
        self.filenames = glob.glob(f"{image_dir}/**/*.jpg", recursive=True)
        self.data_transforms = data_transforms
        self.before_transform = before_transform
        self.after_transform = after_transform

    def __getitem__(self, index):
        img = pil_loader(self.filenames[index])
        img = self.before_transform(img)
        return tuple([self.after_transform(data_transform(img)) for data_transform in self.data_transforms])

    def __len__(self):
        return len(self.filenames)
