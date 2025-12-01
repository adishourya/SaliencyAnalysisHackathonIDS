# from typing import HTML
# only supposed to be used from the notebook
import os
import einops
import torch
from torchvision.io import decode_image
import plotly
import plotly.express as px

import plotly.graph_objects as go
import random
from IPython.display import HTML

import ipywidgets as widgets
from IPython.display import display

random.seed(10)

def run_test(fn1, fn2) -> HTML:
    import numpy as np
    import time
    np.random.seed(int(time.time()))

    from IPython.display import HTML

    html_string = """

     <video alt="test" autoplay=1 height=400 loop>
        <source src="https://openpuppies.com/mp4/{vid}.mp4"  type="video/mp4">
    </video>

        <div style="
        position: absolute;
        top: 10px;
        left: 10px;
        background: rgba(0,0,0,0.5);
        color: {text_color};
        font-size: 20px;
        padding: 6px 12px;
        border-radius: 6px;
    ">
    {msg}
    """
    
    favs=["85Pd59R"]

    happy_pups = ["R8se5g1","eWoEuqz",
                  "6Kmg87X","HlaTE8H",
                  "GjAvXyl","0sa6jrV",
                  "2OU5zUY","ncaCrme",
                  "Hk2qPo3","bRKfspn",
                  "W46GO1L","fiJxCVA",
                  "NQWTWXs","oadIW4Z",
                  "85Pd59R","ruufhZJ",
                  "uRvs9C1"]

    sad_pups = ["DKLBJh7","3V37Hqr","1YraHb7"]

    truth = fn2()
    yours = fn1()
    if yours is None:
        raise Exception("Attempt Before Moving On ?")
    if truth != yours:
        vid_choice = np.random.choice(sad_pups)
        vid_choice = sad_pups[-1]
        return HTML(html_string.format(vid=vid_choice, msg="Try Again",text_color="#bf616a"))
    

    # double check
    assert yours == truth , "Try Again"
    vid_choice = np.random.choice(happy_pups)
    return HTML(html_string.format(vid=vid_choice, msg="Correct!",text_color="#a3be8c"))


# taken from
# Source - https://stackoverflow.com/a
# Posted by Basj, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-30, License - CC BY-SA 4.0

class bidict(dict):
    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key) 

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key) 
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)        

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]: 
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)



def _get_imgpicker_dropdown():
    """
    """
    available_images = []
    for root, dirs, files in os.walk("libs/datasets_and_models/sample_animals10/"):
        for fname in files:
            available_images.append(os.path.join(root, fname))
    random.shuffle(available_images)


    return widgets.Dropdown(
        options=available_images,
        value= "libs/datasets_and_models/sample_animals10/cat/360.jpeg", # cat
        description='choose_image',
        disabled=False,
    )

IMG_PATH_PICKER = _get_imgpicker_dropdown()


CLASS_MAPPING = bidict(
    Butterfly=0,
    Cat=1,
    Chicken=2,
    Cow=3,
    Dog=4,
    Elephant=5,
    Horse=6,
    Sheep=7,
    Spider=8,
    Squirrel=9
)


def plot_picked_img(img):
    img = einops.rearrange(img,"c h w -> h w c")
    fig1 = px.imshow(img,title="Picked Image")
    
    fig1.update_layout(coloraxis_showscale=False)
    fig1.update_xaxes(showticklabels=False)
    fig1.update_yaxes(showticklabels=False)
    fig1.show()
    

def plot_maps(img,sal,title:str,mode_idx:int):
    map_idx = sal[mode_idx,:,:]
    fig = px.imshow(einops.rearrange(img,"c h w -> h w c"),title=title)

    fig.add_trace(
        go.Heatmap(
            z=map_idx,
            colorscale='plasma', # Use a distinct color scale
            opacity=0.6,         # Key: Set transparency (adjust this value!)
            showscale=True,
            name='Saliency'
        )
    )
    
    fig.show()
