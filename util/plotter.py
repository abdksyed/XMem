import cv2
import numpy as np
import matplotlib.pyplot as plt

from dataset.range_transform import inv_im_trans

def plotDatasetSample(dataset, rows):
    first_seq = dataset[0]
    print('rgb', first_seq['rgb'].shape)
    print('first_frame_gt', first_seq['first_frame_gt'].shape)
    print('cls_gt', first_seq['cls_gt'].shape)
    print('selector', first_seq['selector'])
    print('info', first_seq['info'])

    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(10, 15))
    
    for i in range(rows):
        # Plot RGB image
        rgb_image = inv_im_trans(first_seq['rgb'][i]).permute(1,2,0).numpy()
        axes[i, 0].imshow(rgb_image)
        axes[i, 0].axis('off')
        
        # Plot binary mask
        mask = first_seq['cls_gt'][i][0]
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()    


def color_map(pred_mask: np.ndarray, gt_mask: np.ndarray):
    # Intersection of pred_mask and gt_mask: True Positive
    true_positive = np.bitwise_and(pred_mask, gt_mask)
    # Only Pred not GT: False Positive
    false_positive = np.bitwise_and(pred_mask, np.bitwise_not(gt_mask))
    # Only GT not Pred: False Negative
    false_negative = np.bitwise_and(np.bitwise_not(pred_mask), gt_mask)

    # Colors
    green = (0, 255, 0)
    red = (255, 0, 0)
    blue = (0, 0, 255)

    # Creating Color Map Image
    h,w = pred_mask.shape[:2]
    color_map = np.zeros((h,w,3), dtype=np.uint8)
    color_map[true_positive!=0] = green
    color_map[false_positive!=0] = red
    color_map[false_negative!=0] = blue

    return color_map

def frames2video(frames_dict, folder_save_path, video_name, FPS=5):
    video_path = f'{folder_save_path}/{video_name}_{FPS}FPS.mp4'
    print("Creating video and saving:", video_path)
    frame = frames_dict[list(frames_dict.keys())[-1]]
    size1,size2,_ = frame.shape
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (size2, size1), True)
    # Sorting the frames according to frame number eg: frame_007.png
    for _,i in sorted(frames_dict.items(), key=lambda x: x[0]):
        out_img = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        out.write(out_img)
    out.release()