# from typing import HTML
from asyncio import SafeChildWatcher
from IPython.display import HTML
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
