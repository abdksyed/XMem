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