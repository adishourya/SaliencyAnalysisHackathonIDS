from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, random_split # Import random_split
# from cats_dogs_dataloader import train_dataset , val_dataset
from animals10_dataloader import train_dataset, val_dataset
from conv_model import SmallCNN

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import os
from comet_ml import start , ExperimentConfig
from comet_ml.integration.pytorch import log_model,watch


def _save_checkpoint(model):
    print("Saving")
    model_checkpoint = {
            ""
            'model_state_dict': model.state_dict(),
            }
    return model_checkpoint



train_dataloader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True, # Always shuffle the training data
    num_workers=0
)

val_dataloader = DataLoader(
    val_dataset, 
    batch_size=32, 
    shuffle=False, # Do not shuffle validation data (or test data)
    num_workers=0
)



if __name__ == "__main__":
    print("Generating Dataloader")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # for cats_dogs
    # model = SmallCNN(num_classes=37).to(device)

    
    experiment_config = ExperimentConfig(
        log_env_cpu=False,
        log_env_network=False,
        log_env_disk=False,
        auto_histogram_epoch_rate=10,# default 1 
        auto_histogram_weight_logging=True,
    )
    experiment = start(
      api_key=os.getenv("comet_api"),
      project_name="SaliencyIDS",
      workspace="adishourya",
      experiment_config= experiment_config
    )
    experiment_tag = "ConvSaliency"
    experiment.add_tag(experiment_tag)
    # this is costly

    
    
    # for animals 10
    model = SmallCNN(num_classes=10).to(device)
    print(f"{model=}")
    tot_init_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Parameters in model",tot_init_params)
    print("Compiling Model...")
    model = torch.compile(model)
    model.to(device)

    watch(model)
    experiment.log_metric(name="Init Parameters", value=tot_init_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    epochs = 50

    for epoch in range(epochs):
        print(f"Training {epoch=}")
        running_loss = 0.0

        for imgs, labels in tqdm(train_dataloader):
            imgs= imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.4f}")
        experiment.log_metric("TrainingLoss",value=avg_loss,step=epoch)

        with torch.no_grad():
            accuracy = torch.tensor(0.0).to(device)
            val_tot_loss = torch.tensor(0.0).to(device)
            for val_imgs,val_labels in tqdm(val_dataloader):
                val_imgs = val_imgs.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_imgs)
                val_loss = criterion(val_outputs,val_labels)
                val_tot_loss += val_loss.item()
                accuracy+= torch.sum(torch.argmax(val_outputs,dim=1) == val_labels)

            val_tot_loss = val_tot_loss/len(val_dataloader)
            accuracy = accuracy/ len(val_dataset)
            print(f"Validation: Epoch {epoch+1}/{epochs} | Loss: {val_tot_loss:.2f} | Accuracy {accuracy*100}")
            print("=="*10)

        experiment.log_metric("ValidLoss",value=val_tot_loss,step=epoch)
        experiment.log_metric("ValidAccuracy",accuracy,step=epoch)


    checkpoint = _save_checkpoint(model)
    log_model(experiment,checkpoint,model_name=experiment_tag)
    torch.save(model,"checkpoints/conv_model.pth")


