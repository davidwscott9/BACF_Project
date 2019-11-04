def load_video_info(video_path):

    import numpy as np
    import os
    from os import listdir
    file_str = video_path + '/groundtruth_rect.txt'

    lines = open(file_str).read().splitlines()
    ground_truth = np.array(lines[0].split(","))
    for i in range(1, len(lines)):
        row = np.array(lines[i].split(","))
        ground_truth = np.vstack((ground_truth, row))

    ground_truth = ground_truth.astype('float64')

    gt_len = ground_truth.shape[0]
    init_rect = ground_truth[0, :]

    img_path = video_path + '/img/'
    img_files = listdir(img_path)

    seq = {'len': gt_len, 'init_rect': init_rect, 's_frames': img_files}

    return seq, ground_truth
