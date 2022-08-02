import glob
from torch.utils.data import Dataset


class DatasetFolderPairs(Dataset):
    def __init__(self, image_dir, data_transforms, common_transform):
        super(DatasetFolderPairs, self).__init__()
        self.filenames = glob.glob(f"{image_dir}/*.jpg", recursive=True)
        self.data_transforms = data_transforms
        self.common_transform = common_transform

    def __getitem__(self, index):
        return [self.common_transform(data_transform(self.filenames[index])) for data_transform in self.data_transforms]

    def __len__(self):
        return len(self.filenames)
