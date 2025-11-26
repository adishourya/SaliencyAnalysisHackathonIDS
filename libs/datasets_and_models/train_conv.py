from torch.utils.data import DataLoader
from dataloader import cats_dogs_dataset
from conv_model import SmallCNN

import torch
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    print("Generating Dataloader")
    cats_dogs_dataloader = DataLoader(cats_dogs_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    num_classes = len(cats_dogs_dataloader.dataset.classes)
    model = SmallCNN(num_classes).to(device)
    print(f"{model=}")
    print("Compiling...")
    model = torch.compile(model)
    print(f"{num_classes=}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 50

    for epoch in range(epochs):
        print(f"Training {epoch=}")
        running_loss = 0.0

        for imgs, labels in cats_dogs_dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(cats_dogs_dataloader)
        print(f"Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.4f}")
