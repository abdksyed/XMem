import os
import sys
import gc

import typer

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from skimage import io
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from torchmetrics.functional.classification import binary_jaccard_index, multiclass_jaccard_index
from torchmetrics.functional import dice

import wandb

from model.network import XMem

from dataset.range_transform import im_normalization

from inference.inference_core import InferenceCore

from inference.interact.interactive_utils import index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask

from util.plotter import color_map, frames2video

torch.set_grad_enabled(False)

# default configuration
config = {
    'top_k': 30,
    'mem_every': 5,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 128,
    'min_mid_term_frames': 5,
    'max_mid_term_frames': 10,
    'max_long_term_elements': 10000,
}

if torch.cuda.is_available():
  print('Using GPU')
  device = 'cuda'
else:
  print('CUDA not available. Please connect to a GPU instance if possible.')
  device = 'cpu'

torch.cuda.empty_cache()

COLOR = (3, 192, 60)

NUM_OBJECTS = 1 # Binary Segmentation

DATASET_TYPE = "endo"
main_folder = Path("../data")
VIDEOS_PATH = main_folder/"frames"
MASKS_PATH = main_folder/"masks"

test_videos = VIDEOS_PATH
test_masks = MASKS_PATH/"all_masks"


def getIoU(pred_frames, path_dicts, video=True, og_frames = None, save_path=None, num_obj=1):
    
    IoU = torch.zeros(len(pred_frames))
    dice_ = torch.zeros(len(pred_frames))
    overlaid_images = {}
    # get the first mask frame from pred_frames dictionary and get its shape
    _,h,w = pred_frames[list(pred_frames.keys())[0]].shape # [0]:first key
    classes_present = Counter()
    classes_predicted = Counter()
    for i, (frame_name, mask) in enumerate(tqdm(pred_frames.items())):
        np_mask = torch_prob_to_numpy_mask(mask) # predictions probabilities to numpy mask
        classes_predicted.update(Counter(np.unique(np_mask)))
            
        torch_mask = torch.tensor(np_mask).to(device)
        
        truth_mask = io.imread(path_dicts[frame_name][1]) # 0: frame_path, 1: mask_path
        truth_mask[truth_mask == 255] = 1
        if np.sum(truth_mask) < (0.01*h*w): # if mask is empty or covers less than 1% of image
            continue
        classes_present.update(Counter(np.unique(truth_mask)))
        
        if video:
            overlaid_images[frame_name] = cv2.addWeighted(og_frames[frame_name], 1, color_map(np_mask, truth_mask), 0.5, 0)
        if save_path:
            io.imsave(save_path/frame_name, np_mask, check_contrast=False)

        truth_mask = torch.tensor(truth_mask).to(device)

        if num_obj > 1:
            # Have to give background too as num_classes, and in output ignoring background by doing [1:]
            IoU[i] = multiclass_jaccard_index(torch_mask, truth_mask, num_classes=num_obj+1, average='micro', ignore_index=0)
            dice_[i] = dice(torch_mask, truth_mask, num_classes=num_obj+1, average='micro', ignore_index=0)
        else:
            IoU[i] = binary_jaccard_index(torch_mask, truth_mask)
            dice_[i] = dice(torch_mask, truth_mask)


    print("All Present Classes:", classes_present)
    print("All Predicted Classes:", classes_predicted)

    meanIoU = IoU.mean() # Mean across num of frames
    meanDice = dice_.mean()
    
    return meanIoU, IoU, meanDice, dice, overlaid_images


def resize_mask(mask, size, num_obj):
        mask = mask.unsqueeze(0).unsqueeze(0)
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        mask = F.interpolate(mask, (int(h/min_hw*size), int(w/min_hw*size)), 
                    mode='nearest')
        mask = mask.squeeze(0,1).long()
        return F.one_hot(mask, num_classes=num_obj+1).permute(2,0,1).float()

def singleVideoInference(images_paths, first_mask, processor, size = -1, num_obj = 1):
    predictions = {}
    frames = {}
    with torch.cuda.amp.autocast(enabled=True):

        # images_paths = sorted(images_paths)

        # First Frame
        frame = io.imread(images_paths[0])
        og_shape = frame.shape[:2]
        if size < 0:
            im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
            ])
        else:
            im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
                transforms.Resize(size, interpolation=InterpolationMode.BILINEAR, antialias=False),
            ])
            
        frame_torch = im_transform(frame).to(device)
        first_mask = first_mask.astype(np.uint8)
        if size > 0:
            first_mask = torch.tensor(first_mask).to(device)
            first_mask = resize_mask(first_mask, size, num_obj)
        else:
            first_mask = index_numpy_to_one_hot_torch(first_mask, num_obj+1).to(device)
        
        first_mask = first_mask[1:]    
        prediction = processor.step(frame_torch, first_mask)
        
        for image_path in tqdm(images_paths[1:]):
            frame = io.imread(image_path)
            # convert numpy array to pytorch tensor format
            frame_torch = im_transform(frame).to(device)
            
            prediction = processor.step(frame_torch)
            # Upsample to original size if needed
            if size > 0:
                prediction = F.interpolate(prediction.unsqueeze(1), og_shape, mode='bilinear', align_corners=False)[:,0]
            predictions[image_path.name] = prediction
            frames[image_path.name] = frame

    return frames, predictions

def firstMaskGT(mask_files):

    for idx, mask_path in enumerate(mask_files):        
        
        mask = io.imread(mask_path)
        # All 255 Values replaced with 1, other values remain as it is.
        mask = np.where(mask == 255, 1, mask)
        h,w = mask.shape
        if np.sum(mask) > 0 and ( np.sum(mask) > (0.01*h*w) ): # or can use percentage of image, like > 1%
            return mask, idx
            
    return None, -1


def doInference(network_path, config, sorted_paths, size = -1, video=False):
    overallIoU = []
    overallDice = []
    epoch_num = network_path.name.split("_")[-1].split(".")[0]
    for pat_name, sorted_paths_dict in sorted_paths.items():
    
        # Clearing GPU Cache
        torch.cuda.empty_cache()
        network = XMem(config, network_path).eval().to(device)
        processor = InferenceCore(network, config=config)
        NUM_OBJECTS = 11
        processor.set_all_labels(range(1, NUM_OBJECTS+1))

        image_files = [img_path for img_path, _ in sorted_paths_dict.values()]
        mask_files = [mask_path for _, mask_path in sorted_paths_dict.values()]
        
        # Getting first Ground Truth mask.
        mask, start_idx = firstMaskGT(mask_files)
        print("Mask starting from:", start_idx)
    
        print(f"Running Inference on {pat_name}...")
        frames, predictions = singleVideoInference(image_files[start_idx:], mask,
                                                  processor, size = size, num_obj = NUM_OBJECTS)
        save_path = None
        if video:
            save_path = Path(f"./pred_masks/{pat_name}")
            os.makedirs(save_path, exist_ok=True)
            
        IoU, _, dice, _, overlaid_images = getIoU(predictions, sorted_paths_dict,
                                 video=video, og_frames = frames, save_path=save_path,
                                 num_obj = NUM_OBJECTS)
        
        
        print(f"Video \"{pat_name}\", mean IoU is: {IoU*100}")
        wandb.log({pat_name: IoU*100, "epoch": epoch_num})
        print(f"Video \"{pat_name}\", mean dice is: {dice*100}")
        wandb.log({pat_name: dice*100, "epoch": epoch_num})

        # Convert to Video
        if video:
            os.makedirs("./videos", exist_ok=True)
            frames2video(frames_dict=overlaid_images, folder_save_path = "./videos", video_name=pat_name, FPS=5)
        
        overallIoU.append(IoU)
        overallDice.append(dice)
        print()

        del network, processor
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"Average IoU over all videos is: {sum(overallIoU)/len(overallIoU)}.")
    wandb.log({"mIoU": sum(overallIoU)/len(overallIoU), "epoch": epoch_num})
    print(f"Average Dice over all videos is: {sum(overallDice)/len(overallDice)}.")
    wandb.log({"mDice": sum(overallDice)/len(overallDice), "epoch": epoch_num})

    return overallIoU, overallDice

def generate_paths(video_folder_path, mask_folder_path, test_patients=None):
    paths = {}
    for folder in sorted(mask_folder_path.iterdir()):
        pat_name = folder.name if DATASET_TYPE=="endo" else folder.name.split('_')[0]
        if test_patients and pat_name not in test_patients:
            continue
        paths[pat_name] = {}
        for mask_path in folder.iterdir():
            paths[pat_name][mask_path.name] = (video_folder_path/folder.name/mask_path.name, mask_path)
    
    sorted_paths = {pat: {k:v for k,v in sorted(l.items(), key=lambda x: x[0])} for pat, l in paths.items()}

    return sorted_paths

def main():

    runs_map = {
        "RandResize": "plakhsa-mgh/XMem/nxo78a1e",
        "ColorJitter": "plakhsa-mgh/XMem/23eqyeq0",
        "RandAffine": "plakhsa-mgh/XMem/pup4r0wj",
        "RandResizeColor": "plakhsa-mgh/XMem/un32opn7",
        "RandResizeAffine": "plakhsa-mgh/XMem/rd2lhnrw",
        "RandAffineColor": "plakhsa-mgh/XMem/e2r4x4q4",
        "RandResizeColorAffine": "plakhsa-mgh/XMem/499xpw3u"
    }

    test_pat = ["seq_17", "seq_18", "seq_19", "seq_20"]
    TEST_PATIENTS = set([test_pat])
    sorted_paths = generate_paths(test_videos, test_masks, test_patients = TEST_PATIENTS)

    for run_name, path in runs_map.items():
        entity, project, run_id = path.split('/')
        wandb.init(project=project, entity=entity, id=run_id, resume='allow')
        # loop through all pth files in folder f"./augs/{run_name}/saves/"
        for pth_file in Path(f"./augs/{run_name}/saves/").iterdir():
            if not pth_file.suffix == ".pth":
                continue
            network_path = pth_file
            overallIoU, overallDice = doInference(network_path, config, sorted_paths, size = 384)


if __name__ == "__main__":
    # typer.run(main)
    main()