import wandb
import os

runs_map = {
        "RandResize": "plakhsa-mgh/XMem/qae878sk",
        "ColorJitter": "plakhsa-mgh/XMem/0945fdap",
        "RandAffine": "plakhsa-mgh/XMem/d3dcns9o",
        "RandResizeColor": "plakhsa-mgh/XMem/tlfp70jc",
        "RandResizeAffine": "plakhsa-mgh/XMem/mwjb9kxv",
        "RandAffineColor": "plakhsa-mgh/XMem/1d1os0pw",
        "RandResizeColorAffine": "plakhsa-mgh/XMem/dnq7bn44"
    }

for k,v in runs_map.items():
    os.makedirs(f"./augs/{k}", exist_ok=True)
    api = wandb.Api()
    run = api.run(v)
    for i in list(run.files()):
        if "pth" in i.name:
            i.download(f"./augs/{k}")