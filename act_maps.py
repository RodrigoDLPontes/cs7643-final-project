import os
import torch
import pandas as pd
from torch.utils.data import Dataset

class ActivationMaps(Dataset):
    """Feature/activation maps dataset, which need to be generated in the Maps.ipynb notebook."""

    def __init__(self, csv_path="num_cars_corrected.csv",
                 act_map_path="maps", prefix="", split=None):
        """
        Args:
            csv_path (string): Path to the csv file with labels.
            act_map_path (string): Path to folder with feature maps.
            prefix (string): Prefix to be added to all paths.
            split (string): Split to be used, one of "train", "val" or "test". If None, uses all frames.
        """
        csv_path, act_map_path = map(lambda p: os.path.join(prefix, p), (csv_path, act_map_path))

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

        self.act_map_path = act_map_path

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        idx += self.offset

        file_name = f"map{idx + 1:06}.pt"
        file_path = os.path.join(self.act_map_path, file_name)
        act_map = torch.load(file_path).squeeze()

        num_cars = self.labels.loc[idx][0]
        num_cars = torch.tensor(num_cars)

        sample = [ act_map, num_cars ]
        return sample