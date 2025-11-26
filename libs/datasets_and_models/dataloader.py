import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class AnimalsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_names = os.listdir(root_dir)

        # extract labels
        self.labels = ["_".join(name.split(".")[0].split("_")[:-1])
                       for name in self.image_names]

        self.classes = sorted(list(set(self.labels)))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.targets = [self.class_to_idx[lbl] for lbl in self.labels]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.root_dir, img_name)
        # print(img_path)

        # Load with PIL and convert to RGB (handles L, RGBA, etc.)
        img = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        label = self.targets[idx]
        return img, label

# -------------------------------
# Example transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),               # converts to float32 automatically
])

# create dataset
cats_dogs_dataset = AnimalsDataset("./cats_and_dogs_dataset", transform=transform)

if __name__ == "__main__":
    cats_dogs_dataloader = DataLoader(
        cats_dogs_dataset, batch_size=32, shuffle=True, num_workers=0
    )
    for imgs, labels in cats_dogs_dataloader:
        print(imgs.shape, labels.shape)
        break
