
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import os

# --- Configuration ---
# Assuming the root directory is one level above the class folders
DATA_DIR = "/home/adi/code/animals10" 
IMAGE_SIZE = 128
BATCH_SIZE = 32
TRAIN_RATIO = 0.8  # 80% for training
SEED = 42 # For reproducible split

# --- 1. Define Transforms with Augmentation ---
# Aggressive augmentation for small datasets is crucial to prevent overfitting.
# We are incorporating the best practices discussed (Rotation, Affine, Jitter, Flip).
# 

# 1. Training Transforms (WITH Augmentation)
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    
    # 1. Positional Augmentations
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    
    # 2. Color/Pixel Augmentations
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
    transforms.RandomGrayscale(p=0.1), # 10% chance to convert to grayscale
    
    # 3. Final Conversion
    transforms.ToTensor(),
    # Optional but highly recommended: Normalize with ImageNet stats for better model behavior
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 2. Validation Transforms (WITHOUT Augmentation)
# Only resize and convert to tensor.
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Use same as training if you use it
])


# --- 2. Load the Full Dataset using ImageFolder ---
# ImageFolder only needs the root directory; it automatically finds the classes 
# from the subfolder names (butterfly, cat, dog, etc.).
# We apply the Validation transform here, as it's a fixed, non-random transform,
# and we will overwrite the training subset's transform later.
full_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)

# Check if the directory structure was correct
if len(full_dataset.classes) != 10:
    print(f"Warning: Found {len(full_dataset.classes)} classes instead of 10.")
    
print(f"Found {len(full_dataset.classes)} classes: {full_dataset.classes}")

# --- 3. Split the Dataset ---
total_size = len(full_dataset)
train_size = int(total_size * TRAIN_RATIO)
val_size = total_size - train_size

# Split the indices
train_indices, val_indices = random_split(
    full_dataset, 
    lengths=[train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

# --- 4. Apply Correct Transforms to the Subsets ---
# Create the training subset using the training indices
train_dataset = Subset(full_dataset, train_indices.indices)

# Recreate the dataset object specifically for the training set 
# to apply the aggressive train_transform.
# This is a common workaround to apply different transforms to subsets.
train_dataset.dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)

# The validation subset retains the indices and uses the val_transform 
# implicitly set when 'full_dataset' was created.
val_dataset = Subset(full_dataset, val_indices.indices)



# --- 5. Create DataLoaders and Test ---
if __name__ == '__main__':
    if not os.path.isdir(DATA_DIR):
        print(f"Error: Directory '{DATA_DIR}' not found. Please create it and place the 10 class folders inside.")
    else:
        train_ds, val_ds = train_dataset , val_dataset
        
        print("\n--- Data Split Summary ---")
        print(f"Total Images: {len(train_ds) + len(val_ds)}")
        print(f"Training Images: {len(train_ds):,}")
        print(f"Validation Images: {len(val_ds):,}")

        # Create DataLoaders
        train_dataloader = DataLoader(
            train_ds, 
            batch_size=BATCH_SIZE, 
            shuffle=True, # Shuffle training data
            num_workers=0 
        )
        
        val_dataloader = DataLoader(
            val_ds, 
            batch_size=BATCH_SIZE, 
            shuffle=False, # Do NOT shuffle validation data
            num_workers=0
        )

        # Test the DataLoaders
        print("\n--- DataLoader Test ---")
        
        # Training Check
        for images, labels in train_dataloader:
            print(f"Train Batch: Images Shape {images.shape}, Labels Shape {labels.shape}")
            break

        # Validation Check
        for images, labels in val_dataloader:
            print(f"Validation Batch: Images Shape {images.shape}, Labels Shape {labels.shape}")
            break

        # Print Class Mapping
        # Note: To get the class names from a Subset, you need to access the base dataset.
        class_names = train_ds.dataset.classes
        print(f"\nClass Mapping (Index to Name):\n{list(enumerate(class_names))}")
