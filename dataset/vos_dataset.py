import os
from os import path, replace

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization, im_mean # im_mean is (124, 116, 104)
from dataset.reseed import reseed


class VOSDataset(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, im_root, gt_root, max_jump, is_bl, subset=None, num_frames=3, max_num_obj=3, finetune=False):
        self.im_root = im_root # Videos Root Directory
        self.gt_root = gt_root # Ground Truth Root Directory
        self.max_jump = max_jump # Maximum distance between frames, used when sampling
        self.is_bl = is_bl # Whether this is the blender dataset
        self.num_frames = num_frames # Number of frames to sample
        self.max_num_obj = max_num_obj # Maximum number of objects to consider in a frame???

        self.videos = []
        self.frames = {}

        vid_list = sorted(os.listdir(self.im_root)) # List of video names
        # Pre-filtering
        for vid in vid_list:
            if subset is not None:
                if vid not in subset: # If the video is not in the subset, skip it
                    continue
            frames = sorted(os.listdir(os.path.join(self.im_root, vid))) # Sorted list of frames in the video
            if len(frames) < num_frames:
                continue
            self.frames[vid] = frames
            self.videos.append(vid)

        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), im_root))

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.01, 0.01, 0.01, 0),
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0 if finetune or self.is_bl else 15, shear=0 if finetune or self.is_bl else 10, interpolation=InterpolationMode.BILINEAR, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0 if finetune or self.is_bl else 15, shear=0 if finetune or self.is_bl else 10, interpolation=InterpolationMode.NEAREST, fill=0),
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])

        if self.is_bl:
            # Use a different cropping scheme for the blender dataset because the image size is different
            self.all_im_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.25, 1.00), interpolation=InterpolationMode.BILINEAR)
            ])

            self.all_gt_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.25, 1.00), interpolation=InterpolationMode.NEAREST)
            ])
        else:
            self.all_im_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.BILINEAR)
            ])

            self.all_gt_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.NEAREST)
            ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info['name'] = video

        vid_im_path = path.join(self.im_root, video)
        vid_gt_path = path.join(self.gt_root, video)
        frames = self.frames[video]

        trials = 0
        while trials < 5:
            info['frames'] = [] # Appended with actual frames

            num_frames = self.num_frames
            length = len(frames) # eg: 20
            this_max_jump = min(length, self.max_jump) # eg: 3

            # iterative sampling
            frames_idx = [np.random.randint(length)] # Randomly pick a frame eg: 14
            # A->max(0, frames_idx[-1]-this_max_jump): maximum of `0`` or the value of `frames_idx[-1] - this_max_jump`
            # B->min(length, frames_idx[-1]+this_max_jump+1): minimum of `length`` or the value of `frames_idx[-1] + this_max_jump + 1`
            # X->set(range(A,B)): set of numbers starting from A to B (not including B). 
            # Y->set(frames_idx): set of numbers in frames_idx
            # X.difference(Y): Number in X but not in Y 
            # eg: range(11, 17).difference(set(14)) = set(11, 12, 13, 15, 16)
            acceptable_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(length, frames_idx[-1]+this_max_jump+1))).difference(set(frames_idx))
            # This basically makes sure that, the frames selected are between 
            # `-max_jump` and `+max_jump` from any of the frames in frames_idx
            while(len(frames_idx) < num_frames):
                idx = np.random.choice(list(acceptable_set)) # eg: 15
                frames_idx.append(idx) # eg: [14, 15]
                # eg: set(range(12, 18))
                new_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(length, frames_idx[-1]+this_max_jump+1)))
                # eg: set(11, 12, 13, 15, 16, 17).union(set(12,13,14,15,16,17)) -> set(11, 12, 13, 14, 15, 16, 17)
                # set(11, 12, 13, 14, 15, 16, 17).difference(set(14, 15)) = set(11, 12, 13, 16, 17)
                acceptable_set = acceptable_set.union(new_set).difference(set(frames_idx)) 

            frames_idx = sorted(frames_idx) # sort the frames
            if np.random.rand() < 0.5:
                # Reverse time
                frames_idx = frames_idx[::-1]

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            target_objects = []
            for f_idx in frames_idx:
                jpg_name = frames[f_idx][:-4] + '.jpg'
                png_name = frames[f_idx][:-4] + '.png'
                info['frames'].append(jpg_name)

                reseed(sequence_seed)
                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)
                reseed(sequence_seed)
                this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')
                this_gt = self.all_gt_dual_transform(this_gt)

                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im)
                this_im = self.pair_im_lone_transform(this_im)
                reseed(pairwise_seed)
                this_gt = self.pair_gt_dual_transform(this_gt)

                this_im = self.final_im_transform(this_im)
                this_gt = np.array(this_gt)

                images.append(this_im)
                masks.append(this_gt)

            images = torch.stack(images, 0) # (num_frames, 3, 384, 384)

            labels = np.unique(masks[0]) # e.g: [0,1,2,6,9]
            # Remove background
            labels = labels[labels!=0] # e.g: [1,2,6,9]

            if self.is_bl:
                # Find large enough labels
                good_lables = []
                for l in labels:
                    pixel_sum = (masks[0]==l).sum()
                    if pixel_sum > 10*10:
                        # OK if the object is always this small
                        # Not OK if it is actually much bigger
                        if pixel_sum > 30*30:
                            good_lables.append(l)
                        elif max((masks[1]==l).sum(), (masks[2]==l).sum()) < 20*20:
                            good_lables.append(l)
                labels = np.array(good_lables, dtype=np.uint8)
            
            if len(labels) == 0:
                target_objects = []
                trials += 1
            else:
                target_objects = labels.tolist()
                break

        if len(target_objects) > self.max_num_obj:
            target_objects = np.random.choice(target_objects, size=self.max_num_obj, replace=False)

        info['num_objects'] = max(1, len(target_objects))

        masks = np.stack(masks, 0) # (num_frames, 384, 384)

        # Generate one-hot ground-truth
        cls_gt = np.zeros((self.num_frames, 384, 384), dtype=np.int64)
        first_frame_gt = np.zeros((1, self.max_num_obj, 384, 384), dtype=np.int64)
        for i, l in enumerate(target_objects):
            this_mask = (masks==l) # for all frames, get the mask in each frame corresponding to the object `l`
            cls_gt[this_mask] = i+1 # coverting labels from unordered labels to 1,2,...,len(target_objects)
            first_frame_gt[0,i] = (this_mask[0]) # get the mask of the first frame for the object `l`
        cls_gt = np.expand_dims(cls_gt, 1) # (num_frames, 1, 384, 384)

        # 1 if object exist, 0 otherwise
        selector = [1 if i < info['num_objects'] else 0 for i in range(self.max_num_obj)]
        selector = torch.FloatTensor(selector)

        data = {
            'rgb': images,
            'first_frame_gt': first_frame_gt,
            'cls_gt': cls_gt,
            'selector': selector,
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.videos)