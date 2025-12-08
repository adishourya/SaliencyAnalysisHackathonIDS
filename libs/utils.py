# from typing import HTML
# only supposed to be used from the notebook
# 
import numpy as np
import time
import os
import einops
import torch
import torchvision
from torchvision.io import decode_image
import plotly
import plotly.express as px

import plotly.graph_objects as go
import random
from IPython.display import HTML

import ipywidgets as widgets
from IPython.display import display

random.seed(10)

def load_img(path:str):
    img = decode_image(path,mode="RGB")
    # resize the image depending on the input size of the model
    img = einops.rearrange(img,"c h w -> 1 c h w")
    img = torch.nn.functional.interpolate(img,size=(128,128))
    img = img/255.0
    return img

    
    
def run_test(fn1, fn2) -> HTML:
    '''
        umm should i instead take arguments as 2 partials
        and then return bool ?
    '''
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
        vid_choice = np.random.choice(sad_pups)
        return HTML(html_string.format(vid=vid_choice, msg="Try Again",text_color="#bf616a"))
        raise Exception("Attempt Before Moving On ?")
    if truth != yours:
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



def _get_imgpicker_dropdown(choice:int=9):
    # The provided list, stored as a constant
    available_images= [
        'libs/datasets_and_models/sample_animals10/butterfly/OIP-1GLDT_vfdLFEcWBzWaxt7gAAAA.jpeg', 
        'libs/datasets_and_models/sample_animals10/butterfly/OIP-1Sd0G-6ou3Ciuxmfdm3sQAHaHa.jpeg', 
        'libs/datasets_and_models/sample_animals10/butterfly/OIP-3u3iazweX4rxqUcW1ilP-QHaEm.jpeg', 
        'libs/datasets_and_models/sample_animals10/butterfly/OIP-5PFwpG02YpotcjDovzoocwHaE6.jpeg', 
        'libs/datasets_and_models/sample_animals10/butterfly/OIP-KaYXV79KK3t-dKWmB1F5HgHaC_.jpeg', 
        'libs/datasets_and_models/sample_animals10/butterfly/OIP-NGFXrhlEArtgtYOP9ksHEAHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/butterfly/OIP-Pf0pZuRCltIXKUSmK5jayQHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/butterfly/OIP-Pk1BzmOVv2XvCQM14SG5PQHaE6.jpeg', 
        'libs/datasets_and_models/sample_animals10/butterfly/OIP-SAexHo5gbI_VJl-T1tfrkgHaE7.jpeg', 
        'libs/datasets_and_models/sample_animals10/butterfly/OIP-T4jeJE-cUY6CoE8uJ9wS2AHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/butterfly/OIP-TLpb152s1Gouy4zfb4qhEgHaEo.jpeg', 
        'libs/datasets_and_models/sample_animals10/butterfly/OIP-TfY8Ps5q1TcSKeV_-i7TugHaFO.jpeg', 
        'libs/datasets_and_models/sample_animals10/butterfly/OIP-fVa6yA1i4oc4sqGRrgYXCwHaEU.jpeg', 
        'libs/datasets_and_models/sample_animals10/butterfly/OIP-iiHDJqUvQgvSvRNGwz_MLAHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/butterfly/OIP-s5K0u8M2l0xK8HFLPG7EMgHaFi.jpeg', 
        'libs/datasets_and_models/sample_animals10/butterfly/OIP-v-oMb6QiUPaj-IpYpMt38wHaE8.jpeg', 
        'libs/datasets_and_models/sample_animals10/butterfly/e837b90e2cf6093ed1584d05fb1d4e9fe777ead218ac104497f5c97faeebb5bb_640.jpg', 
        'libs/datasets_and_models/sample_animals10/butterfly/ea34b90b29f7053ed1584d05fb1d4e9fe777ead218ac104497f5c97faee8b1b8_640.jpg', 
        'libs/datasets_and_models/sample_animals10/butterfly/eb31b20e2cfd033ed1584d05fb1d4e9fe777ead218ac104497f5c97faeebb5bb_640.jpg', 
        'libs/datasets_and_models/sample_animals10/butterfly/eb3db7062ffd033ed1584d05fb1d4e9fe777ead218ac104497f5c97faee8b1b8_640.jpg', 
        'libs/datasets_and_models/sample_animals10/cat/1261.jpeg', 
        'libs/datasets_and_models/sample_animals10/cat/1307.jpeg', 
        'libs/datasets_and_models/sample_animals10/cat/1494.jpeg', 
        'libs/datasets_and_models/sample_animals10/cat/1502.jpeg', 
        'libs/datasets_and_models/sample_animals10/cat/1556.jpeg', 
        'libs/datasets_and_models/sample_animals10/cat/1621.jpeg', 
        'libs/datasets_and_models/sample_animals10/cat/1771.jpeg', 
        'libs/datasets_and_models/sample_animals10/cat/1784.jpeg', 
        'libs/datasets_and_models/sample_animals10/cat/206.jpeg', 
        'libs/datasets_and_models/sample_animals10/cat/348.jpeg', 
        'libs/datasets_and_models/sample_animals10/cat/360.jpeg', 
        'libs/datasets_and_models/sample_animals10/cat/368.jpeg', 
        'libs/datasets_and_models/sample_animals10/cat/397.jpeg', 
        'libs/datasets_and_models/sample_animals10/cat/419.jpeg', 
        'libs/datasets_and_models/sample_animals10/cat/440.jpeg', 
        'libs/datasets_and_models/sample_animals10/cat/625.jpeg', 
        'libs/datasets_and_models/sample_animals10/cat/675.jpeg', 
        'libs/datasets_and_models/sample_animals10/cat/956.jpeg', 
        'libs/datasets_and_models/sample_animals10/cat/eb34b0062cf0063ed1584d05fb1d4e9fe777ead218ac104497f5c978a7eebdbb_640.jpg', 
        'libs/datasets_and_models/sample_animals10/cat/eb3db30d21f3003ed1584d05fb1d4e9fe777ead218ac104497f5c978a7ebb0bb_640.jpg', 
        'libs/datasets_and_models/sample_animals10/chicken/1041.jpeg', 
        'libs/datasets_and_models/sample_animals10/chicken/126.jpeg', 
        'libs/datasets_and_models/sample_animals10/chicken/251.jpeg', 
        'libs/datasets_and_models/sample_animals10/chicken/422.jpeg', 
        'libs/datasets_and_models/sample_animals10/chicken/571.jpeg', 
        'libs/datasets_and_models/sample_animals10/chicken/OIP-42OMs1eamj9kgtHYh3IQPAAAAA.jpeg', 
        'libs/datasets_and_models/sample_animals10/chicken/OIP-6om1hYGHqnq3Vg3s8waZWwHaGu.jpeg', 
        'libs/datasets_and_models/sample_animals10/chicken/OIP-8Ss8AZETDgyhczkCYltXYQHaD4.jpeg', 
        'libs/datasets_and_models/sample_animals10/chicken/OIP-9e92-_qJlNI8J8gPtcm4swHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/chicken/OIP-CLFdvDJheCKC8cORw7zowAHaJ4.jpeg', 
        'libs/datasets_and_models/sample_animals10/chicken/OIP-Gp3iVb6wQlshSNI8KCVgugHaIf.jpeg', 
        'libs/datasets_and_models/sample_animals10/chicken/OIP-O1OHr-Nts4KxbboCr1-trgHaHa.jpeg', 
        'libs/datasets_and_models/sample_animals10/chicken/OIP-RrttbpaGbBa3FSlyExHtcwHaDx.jpeg', 
        'libs/datasets_and_models/sample_animals10/chicken/OIP-dP3iE5Ev_ytNmxcBergegQAAAA.jpeg', 
        'libs/datasets_and_models/sample_animals10/chicken/OIP-iQFnI1ymR5V_MF2uTO-MswHaFk.jpeg', 
        'libs/datasets_and_models/sample_animals10/chicken/OIP-p-xBTyGIlXd8kVFCfIpHQgHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/chicken/OIP-q5OwO0FfmNNPjf2h0MburwHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/chicken/OIP-sybjvXtgP3b3qWlvQJonlwHaFF.jpeg', 
        'libs/datasets_and_models/sample_animals10/chicken/OIP-woJiqcyQwPwAu29AfnVk8gHaHw.jpeg', 
        'libs/datasets_and_models/sample_animals10/chicken/OIP-zbU3mZZY0gzd95MS38q-fAHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/cow/OIP-4Zc7xV1V9dEJrUzVSq7JvQHaIk.jpeg', 
        'libs/datasets_and_models/sample_animals10/cow/OIP-4ioGYNaH6joXUHrZ_F-s_gAAAA.jpeg', 
        'libs/datasets_and_models/sample_animals10/cow/OIP-5R8bQeX6EuhVQsxaGyuDhgHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/cow/OIP-6NfoHgVIJKfeyaYZh3nwSwHaFO.jpeg', 
        'libs/datasets_and_models/sample_animals10/cow/OIP-Ck5DjiaPLG3cz0eUoX8gCQHaEo.jpeg', 
        'libs/datasets_and_models/sample_animals10/cow/OIP-GRhWXhnZ_u1lX4M9lnCLwQHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/cow/OIP-HqJWFyuU2ph4habfp1QX-QHaE7.jpeg', 
        'libs/datasets_and_models/sample_animals10/cow/OIP-IMEtSmTwPpQ58j11NshumgHaE6.jpeg', 
        'libs/datasets_and_models/sample_animals10/cow/OIP-LkgZow9cK8PteDCI2Yv1ogHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/cow/OIP-MALNGhY7UcPxJoquxLQWlQHaF7.jpeg', 
        'libs/datasets_and_models/sample_animals10/cow/OIP-NkzDid8HU3FTXRQRCuSpgQHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/cow/OIP-QWrJbhjYRr-J407tD2ACUgHaEL.jpeg', 
        'libs/datasets_and_models/sample_animals10/cow/OIP-TwC3ZXBYUigrC-cNId-eZwHaGW.jpeg', 
        'libs/datasets_and_models/sample_animals10/cow/OIP-f0TI4TL2DwemDSImY-JEQgHaE6.jpeg', 
        'libs/datasets_and_models/sample_animals10/cow/OIP-hAFjRVhGoom_XYbQvqJGAAHaE7.jpeg', 
        'libs/datasets_and_models/sample_animals10/cow/OIP-ks0Kv8WEX0tFD_leLmgnAAAAAA.jpeg', 
        'libs/datasets_and_models/sample_animals10/cow/OIP-nopyaGynaYjNqHjVELwZAgHaFv.jpeg', 
        'libs/datasets_and_models/sample_animals10/cow/OIP-qSSi-QP7gifcyXriXIfZYQHaE4.jpeg', 
        'libs/datasets_and_models/sample_animals10/cow/OIP-rVhsW5NLbDPjAKriCxoT0AHaD5.jpeg', 
        'libs/datasets_and_models/sample_animals10/cow/OIP-u5QfAO57j_u7rh9YZUMGwwHaEK.jpeg', 
        'libs/datasets_and_models/sample_animals10/dog/OIP--HjjWS524SecIyPDEh8GNgHaJ4.jpeg', 
        'libs/datasets_and_models/sample_animals10/dog/OIP-2sKsdEI6aaXSSWHJS-QxIAHaJ4.jpeg', 
        'libs/datasets_and_models/sample_animals10/dog/OIP-3HUgRAhJ--2kk9EJuRq05AHaFL.jpeg', 
        'libs/datasets_and_models/sample_animals10/dog/OIP-49TXkdHCI0smpsnvB2Z2rwHaGw.jpeg', 
        'libs/datasets_and_models/sample_animals10/dog/OIP-5OhdwZT0n14XqHpNTV8AAAHaEo.jpeg', 
        'libs/datasets_and_models/sample_animals10/dog/OIP-5RGuxf0vlCJFIa2oVWxOTQAAAA.jpeg', 
        'libs/datasets_and_models/sample_animals10/dog/OIP-GQY8QNtMeVz-ijatA4XkQAHaG_.jpeg', 
        'libs/datasets_and_models/sample_animals10/dog/OIP-JARFRPQT_kT32y55PdCAUAHaE8.jpeg', 
        'libs/datasets_and_models/sample_animals10/dog/OIP-L-ygmvtQbfsac_7jLFyAagHaHh.jpeg', 
        'libs/datasets_and_models/sample_animals10/dog/OIP-LbH_d2FHBnLlpO3L0G-ogwHaFJ.jpeg', 
        'libs/datasets_and_models/sample_animals10/dog/OIP-OszLpf1pm2plOT4SIdL8eQHaEK.jpeg', 
        'libs/datasets_and_models/sample_animals10/dog/OIP-TM3czNnBzNxp3yFJSZZRswHaE7.jpeg', 
        'libs/datasets_and_models/sample_animals10/dog/OIP-_GmIl90tmXb9c72orTGKEQHaJ4.jpeg', 
        'libs/datasets_and_models/sample_animals10/dog/OIP-ekH1fPJXDqNCblUODwjEOQHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/dog/OIP-jC55hR8h_yba45YfCGJk9gHaHa.jpeg', 
        'libs/datasets_and_models/sample_animals10/dog/OIP-rP-TIds3P3w_BmyBu5MYKgHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/dog/OIP-rWYatv8asBcISKlduQPTlgHaE8.jpeg', 
        'libs/datasets_and_models/sample_animals10/dog/OIP-tbDCls3PywwDe98GuL4qswHaFk.jpeg', 
        'libs/datasets_and_models/sample_animals10/dog/OIP-w5jCZcWNfaTz8sM4c3xkHAHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/dog/OIP-xF0nRwKL0hrAGfW0CyP23wHaJQ.jpeg', 
        'libs/datasets_and_models/sample_animals10/elephant/OIP-AIHBjXeB35Js9x4Sa9RW4AHaEK.jpeg', 
        'libs/datasets_and_models/sample_animals10/elephant/OIP-FrJF-hoKI5izIMyCDcMPcwHaE7.jpeg', 
        'libs/datasets_and_models/sample_animals10/elephant/OIP-I4axulCwmqzArS_Hm4ybhgHaE7.jpeg', 
        'libs/datasets_and_models/sample_animals10/elephant/OIP-RH3LwuIcqYaJ3bK8LZaBowHaD8.jpeg', 
        'libs/datasets_and_models/sample_animals10/elephant/OIP-Vg1htIABRj1AZ3tU3i857AHaFQ.jpeg', 
        'libs/datasets_and_models/sample_animals10/elephant/OIP-ViatpK3wPwmHEX4vCFBZVAHaE8.jpeg', 
        'libs/datasets_and_models/sample_animals10/elephant/OIP-X9GqFpai3yaTvy-Y8rF03gHaE7.jpeg', 
        'libs/datasets_and_models/sample_animals10/elephant/OIP-YLAWtT7J2BluyhLcvxdk5AHaEK.jpeg', 
        'libs/datasets_and_models/sample_animals10/elephant/OIP-_2s9bChal44D4sF0wd-pFwAAAA.jpeg', 
        'libs/datasets_and_models/sample_animals10/elephant/OIP-_rEVXpNmSp6MMNJVhilRLgHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/elephant/OIP-b2jQJ-NekdHHdRC2ELN1GQHaE7.jpeg', 
        'libs/datasets_and_models/sample_animals10/elephant/OIP-bC7QB8EGbpO9ZZWA2zsV5gHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/elephant/OIP-bnJqV48i9S43uL-c3XWcyQHaEE.jpeg', 
        'libs/datasets_and_models/sample_animals10/elephant/OIP-fIU1-wtM0sJvoi3-c20C9QHaE8.jpeg', 
        'libs/datasets_and_models/sample_animals10/elephant/OIP-fWSM5Y363wguu8LnbdCI0wHaE8.jpeg', 
        'libs/datasets_and_models/sample_animals10/elephant/OIP-lfbh1jMqzBjNsWtPGvv8gQHaFa.jpeg', 
        'libs/datasets_and_models/sample_animals10/elephant/OIP-qE_Lv17vDdSY8BG_HAZtbgHaEo.jpeg', 
        'libs/datasets_and_models/sample_animals10/elephant/ea36b00e20f1013ed1584d05fb1d4e9fe777ead218ac104497f5c978a4efb4bb_640.jpg', 
        'libs/datasets_and_models/sample_animals10/elephant/ea36b90d2df3083ed1584d05fb1d4e9fe777ead218ac104497f5c978a4efb4bb_640.jpg', 
        'libs/datasets_and_models/sample_animals10/elephant/eb32b40c28f0023ed1584d05fb1d4e9fe777ead218ac104497f5c978a4eebdbd_640.jpg', 
        'libs/datasets_and_models/sample_animals10/horse/OIP-BbRFK5n-WfDwnD1_HL5QhAHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/horse/OIP-EOH0Y6GMn4WUzloQSu-qZAHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/horse/OIP-Fwwzds8k2qsVmd0J_3GKDQHaE6.jpeg', 
        'libs/datasets_and_models/sample_animals10/horse/OIP-JUyzRX-gVFQe3b8PqNy5kgHaGb.jpeg', 
        'libs/datasets_and_models/sample_animals10/horse/OIP-MwwyHSyE__zZIfW4YNmscAHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/horse/OIP-Ofe086_oYegXA8IDXwkoTAHaFc.jpeg', 
        'libs/datasets_and_models/sample_animals10/horse/OIP-PR66G_Ipv0sxTvwrtZIBogHaH5.jpeg', 
        'libs/datasets_and_models/sample_animals10/horse/OIP-_uxyLP5VsLindcLS7zo0JAEsDh.jpeg', 
        'libs/datasets_and_models/sample_animals10/horse/OIP-buCoLEzj0xc_ia_vIm77PgHaFb.jpeg', 
        'libs/datasets_and_models/sample_animals10/horse/OIP-dAqwiSltr_B5ugNk9ETKhQHaFT.jpeg', 
        'libs/datasets_and_models/sample_animals10/horse/OIP-heo9XsjVsOvKB8D1uD9M2gHaGW.jpeg', 
        'libs/datasets_and_models/sample_animals10/horse/OIP-leuAFc3XS_nmvV-jyZiZDgHaE6.jpeg', 
        'libs/datasets_and_models/sample_animals10/horse/OIP-mH_cYFSUpMyu9SYmEXoEfAHaGB.jpeg', 
        'libs/datasets_and_models/sample_animals10/horse/OIP-on4yO75e_cKofrHw2BTxgwHaF7.jpeg', 
        'libs/datasets_and_models/sample_animals10/horse/OIP-p8hsqB3Cr9yFVUoqUwXJwAHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/horse/OIP-s0bCn46ISAPlvk40zjw19wHaFH.jpeg', 
        'libs/datasets_and_models/sample_animals10/horse/OIP-uiS-ZryZUU4FwQrYIYEMigAAAA.jpeg', 
        'libs/datasets_and_models/sample_animals10/horse/OIP-v3vvLDdrTPZrmCQ55EZoowHaEK.jpeg', 
        'libs/datasets_and_models/sample_animals10/horse/OIP-wmGB4TGQKnAieopoFxXllwHaF1.jpeg', 
        'libs/datasets_and_models/sample_animals10/horse/OIP-zwvm9B5IEpwA7LRC45aUsQHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/sheep/OIP-14m0NidrjhJh9b8mSbsNUgHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/sheep/OIP-2SZt2Ice7UQQXqxM2_-_qwHaEo.jpeg', 
        'libs/datasets_and_models/sample_animals10/sheep/OIP-CPXFMldqrcoHAkLw4i8MzwHaDZ.jpeg', 
        'libs/datasets_and_models/sample_animals10/sheep/OIP-H8O7U4ALr6aahPrZveJXugHaF5.jpeg', 
        'libs/datasets_and_models/sample_animals10/sheep/OIP-Hc_6FkdOGKz0pHQsCNV3KwHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/sheep/OIP-ObGlnPV8HOFCnUonD3VFgAHaD_.jpeg', 
        'libs/datasets_and_models/sample_animals10/sheep/OIP-Si2ckV7jsXBkLdcC3BgWBgHaHr.jpeg', 
        'libs/datasets_and_models/sample_animals10/sheep/OIP-TtR2196n7jx_T0txFamXaAHaE8.jpeg', 
        'libs/datasets_and_models/sample_animals10/sheep/OIP-Zoc9SImk5dpVEq-yACfU4AHaE7.jpeg', 
        'libs/datasets_and_models/sample_animals10/sheep/OIP-_00rBolciLYFXcWGfhK7kQHaE7.jpeg', 
        'libs/datasets_and_models/sample_animals10/sheep/OIP-aGzEkEUyJxm_jzxeLSW3JAHaEK.jpeg', 
        'libs/datasets_and_models/sample_animals10/sheep/OIP-baqJKWMvotfDlLnCKj2NxwHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/sheep/OIP-ejzoeuWa4wh2HWkECimT0AHaF6.jpeg', 
        'libs/datasets_and_models/sample_animals10/sheep/OIP-m4UdRn-r2hc83rtyjIQHJAHaGB.jpeg', 
        'libs/datasets_and_models/sample_animals10/sheep/OIP-od77UVNahRWRSy_eyrwibQHaE8.jpeg', 
        'libs/datasets_and_models/sample_animals10/sheep/OIP-s7t85nbWEXh64qNGmJfydAAAAA.jpeg', 
        'libs/datasets_and_models/sample_animals10/sheep/OIP-tQmF0kwe-oKolG8mPJrB3QHaGJ.jpeg', 
        'libs/datasets_and_models/sample_animals10/sheep/e833b10a2dfd073ed1584d05fb1d4e9fe777ead218ac104497f5c978a6eab2b0_640.jpg', 
        'libs/datasets_and_models/sample_animals10/sheep/e836b8062cf7083ed1584d05fb1d4e9fe777ead218ac104497f5c978a6eab2b0_640.jpg', 
        'libs/datasets_and_models/sample_animals10/sheep/eb30b80d2bf4043ed1584d05fb1d4e9fe777ead218ac104497f5c978a6e8b0b1_640.jpg', 
        'libs/datasets_and_models/sample_animals10/spider/OIP--D5FeQbUWqHIGRzsaeSkuQHaE7.jpeg', 
        'libs/datasets_and_models/sample_animals10/spider/OIP-5JuemGqgC7-ImSD1FXV8WAHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/spider/OIP-5Mo2D57HaWIdLWgxnlnzTAHaJF.jpeg', 
        'libs/datasets_and_models/sample_animals10/spider/OIP-7BjvkeeJ0zsPTEbKW_y57QHaLQ.jpeg', 
        'libs/datasets_and_models/sample_animals10/spider/OIP-A9_EBXClbwzEx6C_zGowIAHaE7.jpeg', 
        'libs/datasets_and_models/sample_animals10/spider/OIP-CZ66c8i8bjATPFJQAQ6k5QHaF7.jpeg', 
        'libs/datasets_and_models/sample_animals10/spider/OIP-EGAVUqFTkSOSMYHDJBKfOAHaGw.jpeg', 
        'libs/datasets_and_models/sample_animals10/spider/OIP-HBTFf3yQzyI_NmgHOB3jbQHaFw.jpeg', 
        'libs/datasets_and_models/sample_animals10/spider/OIP-JS1Z-umaK-yACjonR7_oqAAAAA.jpeg', 
        'libs/datasets_and_models/sample_animals10/spider/OIP-JSiO56lEMfelcQicorvqcAHaFU.jpeg', 
        'libs/datasets_and_models/sample_animals10/spider/OIP-cSxJY3iwBmXYMUg6UAt51AHaFS.jpeg', 
        'libs/datasets_and_models/sample_animals10/spider/OIP-fIepTvmNkT-cmqcblbz6yQHaE_.jpeg', 
        'libs/datasets_and_models/sample_animals10/spider/OIP-lB_XcgUtrqIdwg3jgJ1QXwHaFN.jpeg', 
        'libs/datasets_and_models/sample_animals10/spider/OIP-szmk-9ImYlZLlezHdeKXeQHaKJ.jpeg', 
        'libs/datasets_and_models/sample_animals10/spider/OIP-vTgVnZBFqJzWzn8uZPusFQHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/spider/OIP-vZStni-p3tm4BEcc90sKygEsEs.jpeg', 
        'libs/datasets_and_models/sample_animals10/spider/OIP-xFqlOIdRXLdl0gFJtyFUzwHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/spider/e833b5092bf0013ed1584d05fb1d4e9fe777ead218ac104497f5c97ca5edb3bd_640.jpg', 
        'libs/datasets_and_models/sample_animals10/spider/ea36b10f2afd053ed1584d05fb1d4e9fe777ead218ac104497f5c97ca5edb3bd_640.jpg', 
        'libs/datasets_and_models/sample_animals10/spider/eb3cb3082af0073ed1584d05fb1d4e9fe777ead218ac104497f5c97ca5edb3bd_640.png', 
        'libs/datasets_and_models/sample_animals10/squirrel/OIP-1Lfus_MN4kLo85GAPeKDQAHaE8.jpeg', 
        'libs/datasets_and_models/sample_animals10/squirrel/OIP-2TINkS3ll-8tysV3MByFyAHaEy.jpeg', 
        'libs/datasets_and_models/sample_animals10/squirrel/OIP-4m3QwWtPMk8Soh70jRyaKAHaE8.jpeg', 
        'libs/datasets_and_models/sample_animals10/squirrel/OIP-6kQSiyhHF7a-2OYlgpPGhAHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/squirrel/OIP-FV8Fj5JuWMU56i_W-X8NDgHaJ4.jpeg', 
        'libs/datasets_and_models/sample_animals10/squirrel/OIP-GUfvEwhohRRrDm_lN755zAHaLH.jpeg', 
        'libs/datasets_and_models/sample_animals10/squirrel/OIP-KBqSOXL47dieGp2AbN10PAHaGL.jpeg', 
        'libs/datasets_and_models/sample_animals10/squirrel/OIP-N4TkThCX2kNaUeOW0nEBTAHaFU.jpeg', 
        'libs/datasets_and_models/sample_animals10/squirrel/OIP-NWsAybAgTgvyeivPblFrAwHaKq.jpeg', 
        'libs/datasets_and_models/sample_animals10/squirrel/OIP-VXViH2nd5GyiT51uR1Z_hQHaE6.jpeg', 
        'libs/datasets_and_models/sample_animals10/squirrel/OIP-Z3WkMp9I6ReS2K3YnutuqAHaEj.jpeg', 
        'libs/datasets_and_models/sample_animals10/squirrel/OIP-_dVEHaJdv_2qGvKpQOVEJwHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/squirrel/OIP-cc50LzvV_Svt2N7eG1yWmwHaFb.jpeg', 
        'libs/datasets_and_models/sample_animals10/squirrel/OIP-fq4f4c8nMUFl9J0J9XBNSAHaFj.jpeg', 
        'libs/datasets_and_models/sample_animals10/squirrel/OIP-gjdPAAbJl6pxjI6G1Kyj2wHaEK.jpeg', 
        'libs/datasets_and_models/sample_animals10/squirrel/OIP-p_xLwhlFUzseqJ0oUPxybAHaFk.jpeg', 
        'libs/datasets_and_models/sample_animals10/squirrel/OIP-ptDN7t2wsVjztZefwbetOgHaFu.jpeg', 
        'libs/datasets_and_models/sample_animals10/squirrel/OIP-tAeJR_At_Y82kuxuO4tmfwHaFL.jpeg', 
        'libs/datasets_and_models/sample_animals10/squirrel/OIP-vT6pcDuNzdRNiHygWuI9uwHaHa.jpeg', 
        'libs/datasets_and_models/sample_animals10/squirrel/OIP-zEUYFDm5HzhJ7-wwQ84RZAHaE8.jpeg'
    ]

    # this is horrible... and it wont probably work on windows
    # os.walk and sorting does not help.. get 2 diff list on mac and linux(colab)
    # we get 2 diff

    return widgets.Dropdown(
        options=available_images,
        value= available_images[choice],
        # value= "libs/datasets_and_models/sample_animals10/cat/360.jpeg", # cat
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
    if len(img.shape) ==4:
        img =einops.rearrange(img,"1 c h w -> c h w")
    img = einops.rearrange(img,"c h w -> h w c")
    fig1 = px.imshow(img,title="Picked Image")
    
    fig1.update_layout(coloraxis_showscale=False)
    fig1.update_xaxes(showticklabels=False)
    fig1.update_yaxes(showticklabels=False)
    fig1.show()
    

def plot_maps(img,sal,title:str,mode_idx:int):
    if len(sal.shape) == 2:
        sal = sal[None,:,:]
    map_idx = sal[mode_idx,:,:]

    # map_idx = torchvision.transforms.functional.gaussian_blur(map_idx,31)
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
