import os
from os import path

from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset

from torchvision import transforms

from dataset.range_transform import im_normalization
from dataset.reseed import reseed

class SurgDataset(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, im_root, gt_root, max_jump,
                 im_tran=transforms.Compose([]),
                 imgt_tran=[transforms.Compose([]), transforms.Compose([])],
                 all_im=transforms.Compose([]),
                 all_imgt=[transforms.Compose([]), transforms.Compose([])],
                 subset=None, num_frames=3, max_num_obj=1,
                 finetune=False):
        
        self.im_root = im_root # Root directory for Images
        self.gt_root = gt_root # Root directory for ground truth data
        self.max_jump = max_jump # Maximum distance between frames
        # self.is_bl = is_bl # Is it a blender dataset
        self.num_frames = num_frames # Number of frames to be sampled
        self.max_num_obj = max_num_obj # Maximum number of objects

        # Initialize lists for storing video and frame information
        self.videos = [] # List of videos
        self.frames = {} # Dictionary mapping video to its frames

        vid_list = sorted(os.listdir(self.im_root))
        # Pre-filtering
        for vid in vid_list:
            # If looking for a subset of videos only
            if subset is not None:
                vid_name = vid
                if vid_name not in subset:
                    continue
            
            # List frames in each video directory
            frames = sorted(os.listdir(os.path.join(self.im_root, vid)))
            
            if len(frames) < num_frames:
                continue
            self.frames[vid] = frames
            self.videos.append(vid)

        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), im_root))

        self.im_tran = im_tran
        self.imgt_tran = imgt_tran
        self.all_im = all_im
        self.all_imgt = all_imgt

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
            length = len(frames)
            this_max_jump = min(len(frames), self.max_jump)

            # iterative sampling
            frames_idx = [np.random.randint(length)]
            acceptable_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(length, frames_idx[-1]+this_max_jump+1))).difference(set(frames_idx))
            while(len(frames_idx) < num_frames):
                idx = np.random.choice(list(acceptable_set))
                frames_idx.append(idx)
                new_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(length, frames_idx[-1]+this_max_jump+1)))
                acceptable_set = acceptable_set.union(new_set).difference(set(frames_idx))

            frames_idx = sorted(frames_idx)
            if np.random.rand() < 0.5:
                # Reverse time
                frames_idx = frames_idx[::-1]

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            target_objects = []
            for f_idx in frames_idx:
                jpg_name = frames[f_idx]
                png_name = frames[f_idx]
                info['frames'].append(jpg_name)

                reseed(sequence_seed)
                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
                this_im = self.all_imgt[0](this_im)
                this_im = self.all_im(this_im)
                reseed(sequence_seed)
                this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')
                this_gt = self.all_imgt[1](this_gt)

                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.imgt_tran[0](this_im)
                this_im = self.im_tran(this_im)
                reseed(pairwise_seed)
                this_gt = self.imgt_tran[1](this_gt)
                
                this_im = self.final_im_transform(this_im)
                this_gt = np.array(this_gt)

                images.append(this_im)
                masks.append(this_gt)

            images = torch.stack(images, 0)

            labels = np.unique(masks[0])
            # Remove background
            labels = labels[labels!=0]

            
            if len(labels) == 0:
                target_objects = []
                trials += 1
            else:
                target_objects = labels.tolist()
                break

        if len(target_objects) > self.max_num_obj:
            target_objects = np.random.choice(target_objects, size=self.max_num_obj, replace=False)

        info['num_objects'] = max(1, len(target_objects))

        masks = np.stack(masks, 0)

        # Generate one-hot ground-truth
        cls_gt = np.zeros((self.num_frames, 384, 384), dtype=np.int64)
        first_frame_gt = np.zeros((1, self.max_num_obj, 384, 384), dtype=np.int64)
        for i, l in enumerate(target_objects):
            this_mask = (masks==l)
            cls_gt[this_mask] = i+1
            first_frame_gt[0,i] = (this_mask[0])
        cls_gt = np.expand_dims(cls_gt, 1)

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