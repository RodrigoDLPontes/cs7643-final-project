import os
import torch
import numpy as np
import pandas as pd
from skimage import io
from torch.utils.data import Dataset

class GRAM_RTM(Dataset):
    """GRAM Road-Traffic Monitoring (GRAM-RTM) dataset."""

    def __init__(self, csv_path="num_cars_corrected.csv", img_path="M-30", roi_path="ROI.jpg", prefix=""):
        """
        Args:
            csv_path (string): Path to the csv file with labels.
            root_path (string): Path to folder with all the M-30 frames.
            roi_path (string): Path to ROI mask.
            prefix (string): Prefix to be added to all paths.
        """
        csv_path, img_path, roi_path = map(lambda p: os.path.join(prefix, p), (csv_path, img_path, roi_path))
        self.labels = pd.read_csv(csv_path)
        self.img_path = img_path
        roi = io.imread(roi_path)
        self.roi = roi // 255

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        file_name = f"image{idx + 1:06}.jpg"
        file_path = os.path.join(self.img_path, file_name)
        image = io.imread(file_path)
        masked_img = image * self.roi
        num_cars = self.labels.iloc[idx, 0]
        num_cars = np.array([num_cars])
        sample = {'image': masked_img, 'num_cars': num_cars}

        return sample