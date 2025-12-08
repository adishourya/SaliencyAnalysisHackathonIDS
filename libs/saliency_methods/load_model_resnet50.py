import os
import torch
import torch.nn as nn
import torchvision.models as models # NEW: Import for pre-trained models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH= os.path.join(
    BASE_DIR,
    "../datasets_and_models/checkpoints/ResNet50_Transfer_state_dict.pth"
)



try:
    from libs.datasets_and_models.animals10_dataloader import val_dataset
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, # Do not shuffle validation data (or test data)
        num_workers=0
    )
except FileNotFoundError:
    pass

def get_resnet():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    state_dict = torch.load(CHECKPOINT_PATH,"cpu",weights_only=True)
    # print(state_dict)
    # state_dict = state_dict["model_state_dict"]

    new_state_dict = dict()
    for keys in state_dict.keys():
        new_key_name = "".join(keys.split("_mod.")[1:])
        new_state_dict[new_key_name] = state_dict[keys]

    model.load_state_dict(new_state_dict)
    model.eval()
    return model

    
def test_model(model):
    print("Performing Eval Time Inference")
    with torch.no_grad():
        accuracy =torch.tensor(0.0)
        for imgs,label in tqdm(val_dataloader):
            out=model(imgs)
            out = torch.argmax(out,dim=-1)
            accuracy += torch.sum(out==label)
        accuracy = accuracy/len(val_dataset) * 100

    print(f"Accuracy Inference Time:{accuracy}")

if __name__ == "__main__":
    model = get_resnet()
    test_model(model)
    
