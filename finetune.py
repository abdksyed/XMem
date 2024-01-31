from os import path
import math
from pathlib import Path
from dataclasses import asdict

import typer

import random
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.distributed as distributed
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import wandb

from model.trainer import XMemTrainer
from dataset import SurgDataset, im_mean

from util.plotter import plotDatasetSample
from util.logger import TensorboardLogger
from util.configuration import Config


"""
Initial setup
"""

def main(exp_id:str):

    # Init distributed environment
    distributed.init_process_group(backend="nccl")
    print(f"CUDA Device count: {torch.cuda.device_count()}")

    # Load configuration
    config = Config()
    config.exp_id = exp_id
    config.max_num_obj = 1
    wandb.init(project="XMem", name=config.exp_id, config=config)

    if config.benchmark:
        torch.backends.cudnn.benchmark = True

    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    torch.cuda.set_device(local_rank)

    print(f"I am rank {local_rank} in this world of size {world_size}!")

    torch.manual_seed(14159265)
    np.random.seed(14159265)
    random.seed(14159265)

    main_folder = Path("../data")
    VIDEOS_PATH = main_folder / "frames"
    MASKS_PATH = main_folder / "masks"

    train_videos = VIDEOS_PATH
    train_masks = MASKS_PATH / "all_masks"

    sample_dataset = SurgDataset(
        train_videos,
        train_masks,
        max_jump=10,
        all_imgt = [
            transforms.Compose([transforms.Resize((384,384))]),
            transforms.Compose([transforms.Resize((384,384))])
            ],
        subset={"seq_20"},
        num_frames=8,
        max_num_obj=config.max_num_obj,
        finetune=False,
    )
    f = plotDatasetSample(sample_dataset, 5)

    git_info = 'XMem'
    id = config.exp_id
    logger = TensorboardLogger(git_info, id)
    logger.log_string('hyperpara', str(asdict(config)))

    model = XMemTrainer(
        asdict(config),
        logger=logger,
        save_path=path.join("saves", config.exp_id),
        local_rank=local_rank,
        world_size=world_size,
    ).train()

    model.load_network("./saves/XMem.pth")
    total_iter = 0

    # Transformations
    if "Color" in exp_id:
        im_train = transforms.Compose(
            [
                transforms.ColorJitter(0.25, 0.1, 0.1, 0),
            ]
        )
    else:
        im_train = transforms.Compose([])
    
    if "Affine" in exp_id:
        imgt_tran = [
            transforms.Compose(
                [
                    transforms.RandomAffine(
                        degrees=15,
                        shear=10,
                        interpolation=InterpolationMode.BILINEAR,
                        fill=im_mean,
                    ),
                ]
            ),
            transforms.Compose(
                [
                    transforms.RandomAffine(
                        degrees=15, shear=10, interpolation=InterpolationMode.NEAREST, fill=0
                    ),
                ]
            ),
        ]
    else:
        imgt_tran = [transforms.Compose([]), transforms.Compose([])]

    if False: # Add some Image only exact same Augmentations for all images in video sequence
        all_im = transforms.Compose(
            [
                transforms.ColorJitter(0.1, 0.03, 0.03, 0),
                transforms.RandomGrayscale(0.05),
            ]
        )
    else:
        all_im = transforms.Compose([])

    if "Resize" in exp_id:
        all_imgt = [
            transforms.Compose(
                [
                    # transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(
                        (384, 384), scale=(0.36, 1.00), interpolation=InterpolationMode.BILINEAR
                    ),
                ]
            ),
            transforms.Compose(
                [
                    # transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(
                        (384, 384), scale=(0.36, 1.00), interpolation=InterpolationMode.NEAREST
                    ),
                ]
            ),
        ]
    else:
        all_imgt = [
            transforms.Compose([transforms.Resize((384,384))]),
            transforms.Compose([transforms.Resize((384,384))])
            ]
        

    # To re-seed the randomness everytime we start a worker
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % (2**31) + worker_id + local_rank * 100
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    def renew_loader(
        video_path,
        masks_path,
        max_skip,
        subset,
        max_num_obj=config.max_num_obj,
        finetune=True,
    ):
        dataset = SurgDataset(
            video_path,
            masks_path,
            max_jump=max_skip,
            im_tran = im_train,
            imgt_tran = imgt_tran,
            all_im = all_im,
            all_imgt = all_imgt,
            subset=subset,
            num_frames=config.num_frames,
            max_num_obj=max_num_obj,
            finetune=finetune,
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, rank=local_rank, shuffle=True
        )
        train_loader = DataLoader(
            dataset,
            config.batch_size,
            sampler=train_sampler,
            num_workers=config.num_workers,
            worker_init_fn=worker_init_fn,
            drop_last=True,
            pin_memory=True,
        )

        return train_sampler, train_loader

    MAX_SKIP_VALUES = [10, 15, 5, 5]

    increase_skip_fraction = [0.1, 0.3, 0.9, 100]

    TEST_PATIENTS = set(["p05", "p11"])
    train_subset = set()
    for folder in train_videos.iterdir():
        pat_name = folder.name
        if not pat_name in TEST_PATIENTS:
            train_subset.add(folder.name)

    train_sampler, train_loader = renew_loader(train_videos, train_masks, 20, subset = train_subset, finetune=False)

    total_epoch = math.ceil(config.iterations/len(train_loader))
    current_epoch = total_iter // len(train_loader)
    print(f'Current epoch is {current_epoch}.')
    print(f'We approximately use {total_epoch} epochs.')

    change_skip_iter = [round(config.iterations*f) for f in increase_skip_fraction]
    # Skip will only change after an epoch, not in the middle
    print(f'The skip value will change approximately at the following iterations: {change_skip_iter[:-1]}')

    np.random.seed(np.random.randint(2**30-1) + local_rank*100)
    try:
        while total_iter < config.iterations:
            
            # Crucial for randomness! 
            train_sampler.set_epoch(current_epoch)
            current_epoch += 1
            print(f'Current epoch: {current_epoch}')

            # Train loop
            model.train()
            for data in train_loader:
                # Update skip if needed
                if total_iter >= change_skip_iter[0]:
                    while total_iter >= change_skip_iter[0]:
                        cur_skip = MAX_SKIP_VALUES[0]
                        max_skip_values = MAX_SKIP_VALUES[1:]
                        change_skip_iter = change_skip_iter[1:]
                    print(f'Changing skip to {cur_skip=}')
                    train_sampler, train_loader = renew_loader(train_videos, train_masks, cur_skip,
                                                            subset = train_subset, finetune=False)
                    break

                model.do_pass(data, total_iter)
                total_iter += 1

                if total_iter >= config.iterations:
                    break
    finally:
            model.save_network(total_iter)
            print("Done")

    distributed.destroy_process_group()


if __name__ == "__main__":
    typer.run(main)