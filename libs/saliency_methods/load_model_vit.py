import os
import torch
import torch.nn as nn
import torchvision.models as models # NEW: Import for pre-trained models
from tqdm import tqdm
from libs.datasets_and_models.animals10_dataloader import val_dataset
from torch.utils.data import Dataset, DataLoader


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH= os.path.join(
    BASE_DIR,
    "../datasets_and_models/checkpoints/ViT_Transfer_state_dict.pth"
)


val_dataloader = DataLoader(
    val_dataset, 
    batch_size=32, 
    shuffle=False, # Do not shuffle validation data (or test data)
    num_workers=0
)

def get_vit():
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT) 
    num_ftrs = model.heads.head.in_features
    NUM_CLASSES = 10 # Since you are using animals 10
    model.heads.head = nn.Linear(num_ftrs, NUM_CLASSES)

    state_dict = torch.load(CHECKPOINT_PATH,"cpu",weights_only=True)
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
    model = get_vit()
    test_model(model)
    
