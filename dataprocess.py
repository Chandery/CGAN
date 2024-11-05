import os
from PIL import Image
from torchvision.transforms import ToTensor, Normalize
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.image_files = os.listdir(directory)
        self.ToTensor = ToTensor()
        self.Norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.directory, image_file)
        image = Image.open(image_path)
        return self.Norm(self.ToTensor(image))

if __name__ == '__main__':
    dataset = ImageDataset('data/comic_faces_dataset')
    print(len(dataset))
    print(dataset.shape)