import wandb
import os

runs_map = {
        "RandResize": "plakhsa-mgh/XMem/nxo78a1e",
        "ColorJitter": "plakhsa-mgh/XMem/23eqyeq0",
        "RandAffine": "plakhsa-mgh/XMem/pup4r0wj",
        "RandResizeColor": "plakhsa-mgh/XMem/un32opn7",
        "RandResizeAffine": "plakhsa-mgh/XMem/rd2lhnrw",
        "RandAffineColor": "plakhsa-mgh/XMem/e2r4x4q4",
        "RandResizeColorAffine": "plakhsa-mgh/XMem/499xpw3u"
    }

for k,v in runs_map.items():
    os.makedirs(f"./augs/{k}", exist_ok=True)
    api = wandb.Api()
    run = api.run(v)
    for i in list(run.files()):
        if "pth" in i.name:
            i.download(f"./augs/{k}")