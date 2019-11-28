from load_video_info import load_video_info
from run_BACF import run_BACF
from computePascalScore import computePascalScore
import numpy as np
from os import listdir
import matplotlib.pyplot as plt

# MIGHT BE BETTER TO MAKE THIS A GENERAL TESTING FUNCTION WITH PARAMS(str('TEST'))
# HAVE IF STATEMENTS FOR WHAT TO DO FOR EACH TEST TYPE

# Load video information
base_path = './seq/'
test_type = 'OTB50'  # MIGHT MAKE THIS INTO A FUNCTION AND OTB50 vs 100 IS DETERMINED IN THE SCRIPT

test_path = base_path + test_type
video_list = listdir(test_path)

# Remove videos with incomplete ground-truths
video_list.remove('David')
video_list.remove('Diving')
video_list.remove('Freeman4')

overlap_threshold = np.arange(0, 1, 0.1) + 0.1
OP_vid = []
for video in video_list:
    print(video)
    video_path = test_path + '/' + video
    [seq, ground_truth] = load_video_info(video_path)
    seq['VidName'] = video
    seq['st_frame'] = 0
    seq['en_frame'] = seq['len']

    # Contingencies for videos with incomplete groundtruths
    if video == 'David':
        seq['st_frame'] = 299
        seq['en_frame'] = 770
        continue
    elif video == 'Diving':
        continue
    # elif video == 'Football':
    #     seq['st_frame'] = 0
    #     seq['en_frame'] = 74
    elif video == 'Freeman4':
        seq['st_frame'] = 0
        seq['en_frame'] = 282
        continue

    gt_boxes = np.concatenate([ground_truth[:, 0:2],
                               ground_truth[:, 0:2] + ground_truth[:, 2:4] - np.ones([ground_truth.shape[0], 2])],
                              axis=1)

    # Run BACF main function
    learning_rate = 0.013  # MAYBE ADD THIS AS A PARAMETER IN THE FUNCTION
    results = run_BACF(seq, video_path, learning_rate, visualization=0)

    # compute the OP
    pd_boxes = results['res']
    pd_boxes = np.concatenate([pd_boxes[:, 0:2],
                               pd_boxes[:, 0:2] + pd_boxes[:, 2:4] - np.ones([pd_boxes.shape[0], 2])], axis=1)
    OP = np.zeros([gt_boxes.shape[0], 1])

    for i in range(0, gt_boxes.shape[0]):
        b_gt = gt_boxes[i, :]
        b_pd = pd_boxes[i, :]
        OP[i] = computePascalScore(b_gt, b_pd)

    if len(OP_vid) == 0:
        OP_vid = np.sum(OP >= overlap_threshold, axis=0) / len(OP)
        FPS_vid = results['fps']
    else:
        OP_vid = np.vstack((OP_vid, np.sum(OP >= overlap_threshold, axis=0) / len(OP)))
        FPS_vid = np.vstack((FPS_vid, results['fps']))

OP_vid_avg = np.mean(OP_vid, axis=0)

fig, ax = plt.subplots(1,2)
for j in range(0, len(video_list)):
    ax[0].plot(overlap_threshold, OP_vid[j, :], label=video_list[j])
ax[0].legend()
ax[0].set_xlabel('Overlap Threshold')
ax[0].set_ylabel('Success Rate')

ax[1].plot(overlap_threshold, OP_vid_avg, label='Average Result')
ax[1].legend()
ax[1].set_xlabel('Overlap Threshold')
ax[1].set_ylabel('Success Rate')
title_str = 'Success Plot on ' + test_type
fig.suptitle(title_str)

