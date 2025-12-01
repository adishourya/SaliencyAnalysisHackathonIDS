import torch
from tqdm import tqdm
from libs.datasets_and_models.conv_model import SmallCNN
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONV_CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "../datasets_and_models/checkpoints/model-data_comet-torch-model-49.pth"
)


def get_conv_model():
    model = SmallCNN(10)
    state_dict = torch.load(CONV_CHECKPOINT_PATH,"cpu",weights_only=True)
    state_dict = state_dict["model_state_dict"]

    new_state_dict = dict()
    for keys in state_dict.keys():
        new_key_name = "".join(keys.split("_mod.")[1:])
        new_state_dict[new_key_name] = state_dict[keys]

    model.load_state_dict(new_state_dict)
    model.eval()
    return model

    
def test_model(model):

    from libs.datasets_and_models.animals10_dataloader import val_dataset
    from torch.utils.data import Dataset, DataLoader
    print("Performing Eval Time Inference")

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, # Do not shuffle validation data (or test data)
        num_workers=0
    )
    with torch.no_grad():
        accuracy =torch.tensor(0.0)
        for imgs,label in tqdm(val_dataloader):
            out=model(imgs)
            out = torch.argmax(out,dim=-1)
            accuracy += torch.sum(out==label)
        accuracy = accuracy/len(val_dataset) * 100

    print(f"Accuracy Inference Time:{accuracy}")
