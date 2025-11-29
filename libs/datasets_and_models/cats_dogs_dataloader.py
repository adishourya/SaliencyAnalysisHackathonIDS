import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms

# --- AnimalsDataset Class (Unchanged) ---
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
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.targets[idx]
        return img, label

# ----------------------------------------------------
# 1. SETUP SEPARATE TRANSFORMS FOR TRAINING AND VALIDATION
# ----------------------------------------------------

IMAGE_SIZE = 128

# Augmentation for the Training Set (Adding randomness and noise)
train_transform = transforms.Compose([
    # Resize the image first
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    
    # 1. Essential: Randomly flip the image
    transforms.RandomHorizontalFlip(p=0.5), 
    
    # 2. Strong positional variance: Randomly crops and resizes
    # You had this already, good!
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    
    # 3. Essential: Adjust brightness, contrast, and saturation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
    
    # 4. Convert to Tensor and normalize
    transforms.ToTensor(), 
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Optional, but recommended later
    transforms.RandomRotation(15),
    transforms.RandomAffine(0,(0.1,0.1)),
    transforms.RandomGrayscale(),
    transforms.GaussianBlur(3),
])

# Deterministic Transforms for the Validation Set (No randomness!)
# We only want to ensure the image is the correct size and format.
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Optional, use same as training
])


# ----------------------------------------------------
# 2. DATASET AND DATALOADER CREATION (Revised)
# ----------------------------------------------------

DATA_DIR = "/home/adi/code/cats_dogs"

# Create the full dataset without any specific transform first
full_dataset = AnimalsDataset(DATA_DIR)

# --- Splitting the Dataset ---
DATASET_SIZE = len(full_dataset)
TRAIN_RATIO = 0.8

train_length = int(DATASET_SIZE * TRAIN_RATIO)
val_length = DATASET_SIZE - train_length

# Perform the random split (This splits the INDICES)
train_indices, val_indices = random_split(
    full_dataset, 
    lengths=[train_length, val_length],
    generator=torch.Generator().manual_seed(42)
)

# --- Apply Transforms to Subsets ---
# We now wrap the subsets to apply the correct transforms
train_dataset = AnimalsDataset(DATA_DIR, transform=train_transform)
train_dataset.image_names = [full_dataset.image_names[i] for i in train_indices.indices]
train_dataset.targets = [full_dataset.targets[i] for i in train_indices.indices]

val_dataset = AnimalsDataset(DATA_DIR, transform=val_transform)
val_dataset.image_names = [full_dataset.image_names[i] for i in val_indices.indices]
val_dataset.targets = [full_dataset.targets[i] for i in val_indices.indices]

print(f"Total images: {DATASET_SIZE}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")


if __name__ == "__main__":
    # 3. Create DataLoaders
    BATCH_SIZE = 32

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0
    )

    # Example iteration over the training DataLoader
    print("\nTraining DataLoader check:")
    for imgs, labels in train_dataloader:
        print(f"Batch shape: {imgs.shape}, Labels shape: {labels.shape}")
        break

    # Example iteration over the validation DataLoader
    print("\nValidation DataLoader check:")
    for imgs, labels in val_dataloader:
        print(f"Batch shape: {imgs.shape}, Labels shape: {labels.shape}")
        break
