def load_video_info(video_path, test_type):

    import numpy as np
    import os
    from os import listdir

    if test_type == 'TC128':
        files = listdir(video_path)
        gt = [s for s in files if "gt.txt" in s][0]
        file_str = video_path + '/' + gt

    else:
        file_str = video_path + '/groundtruth_rect.txt'

    lines = open(file_str).read().splitlines()
    ground_truth = np.array(lines[0].split(","))
    # length should be 4. If not, didn't separate data properly. Try another delimiter type
    if len(ground_truth) == 1:
        ground_truth = np.array(lines[0].split("\t"))  # in case tab delimited values instead of comma
    for i in range(1, len(lines)):
        row = np.array(lines[i].split(","))
        # length should be 4. If not, didn't separate data properly. Try another delimiter type
        if len(row) == 1:
            row = np.array(lines[i].split("\t"))  # in case tab delimited values instead of comma
        ground_truth = np.vstack((ground_truth, row))

    ground_truth = ground_truth.astype('float64')
    gt_len = ground_truth.shape[0]
    init_rect = ground_truth[0, :]

    img_path = video_path + '/img/'
    img_files = listdir(img_path)

    seq = {'len': gt_len, 'init_rect': init_rect, 's_frames': img_files}

    return seq, ground_truth
