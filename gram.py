import os
import PIL
import torch
import pandas as pd
from skimage import io
from torchvision import transforms
from torch.utils.data import Dataset

class GRAM_RTM(Dataset):
    """GRAM Road-Traffic Monitoring (GRAM-RTM) dataset."""

    def __init__(self, csv_path="num_cars_corrected.csv", img_path="M-30", roi_path="ROI.jpg",
                 prefix="", split=None, spec_nn=False, detectron=False):
        """
        Args:
            csv_path (string): Path to the csv file with labels.
            root_path (string): Path to folder with all the M-30 frames.
            roi_path (string): Path to ROI mask.
            prefix (string): Prefix to be added to all paths.
        """
        csv_path, img_path, roi_path = map(lambda p: os.path.join(prefix, p), (csv_path, img_path, roi_path))

        all_labels = pd.read_csv(csv_path)
        num_train = int(len(all_labels) * 0.6)
        num_val_test = int(len(all_labels) * 0.2)
        if split == "train":
            self.labels = all_labels[:num_train]
            self.offset = 0
        elif split == "val":
            self.labels = all_labels[num_train:num_train + num_val_test]
            self.offset = num_train
        elif split == "test":
            self.labels = all_labels[num_train + num_val_test:]
            self.offset = num_train + num_val_test
        else:
            self.labels = all_labels
            self.offset = 0

        self.img_path = img_path

        roi = io.imread(roi_path)
        self.spec_nn = spec_nn
        self.detectron = detectron
        if self.detectron:
            self.roi = roi // 255
        else:
            roi = torch.tensor(roi[:,:,0] // 255).unsqueeze(0)
            if self.spec_nn:
                self.roi = transforms.Resize((65, 65), PIL.Image.NEAREST)(roi)
            else:
                self.roi = transforms.Resize(224, PIL.Image.NEAREST)(roi)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        idx += self.offset

        file_name = f"image{idx + 1:06}.jpg"
        file_path = os.path.join(self.img_path, file_name)
        image = io.imread(file_path)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self.detectron:
            image = image * self.roi
        else:
            image = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])(image)
            if self.spec_nn:
                image = transforms.Resize((65, 65))(image)
            else:
                image = transforms.Resize(224)(image)
            image = image * self.roi

        num_cars = self.labels.loc[idx][0]
        num_cars = torch.tensor(num_cars)

        sample = [ image, num_cars ]
        return sample