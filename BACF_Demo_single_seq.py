from load_video_info import load_video_info
from run_BACF import run_BACF
from computePascalScore import computePascalScore
import numpy as np

# Load video information
base_path = './seq/OTB50'
video = 'Bolt'
video_path = base_path + '/' + video
[seq, ground_truth] = load_video_info(video_path, test_type='OTB50')
seq['VidName'] = video
seq['st_frame'] = 1
seq['en_frame'] = seq['len']

# Contingencies for videos with incomplete groundtruths
if video == 'David':
    seq['st_frame'] = 299
    seq['en_frame'] = 770
elif video == 'Diving':
    seq['st_frame'] = 0
    seq['en_frame'] = 214
elif video == 'Football':
    seq['st_frame'] = 0
    seq['en_frame'] = 74
elif video == 'Freeman3':
    seq['st_frame'] = 0
    seq['en_frame'] = 460
elif video == 'Freeman4':
    seq['st_frame'] = 0
    seq['en_frame'] = 282

gt_boxes = np.concatenate([ground_truth[:, 0:2],
                           ground_truth[:, 0:2] + ground_truth[:, 2:4] - np.ones([ground_truth.shape[0], 2])], axis=1)

# Run BACF main function
learning_rate = 0.013
results = run_BACF(seq, video_path, learning_rate, visualization=1)

# compute the OP
pd_boxes = results['res']
pd_boxes = np.concatenate([pd_boxes[:, 0:2],
                           pd_boxes[:, 0:2] + pd_boxes[:, 2:4] - np.ones([pd_boxes.shape[0], 2])], axis=1)
OP = np.zeros([gt_boxes.shape[0], 1])

for i in range(0, gt_boxes.shape[0]):
    b_gt = gt_boxes[i, :]
    b_pd = pd_boxes[i, :]
    OP[i] = computePascalScore(b_gt, b_pd)

OP_vid = np.sum(OP >= 0.5) / len(OP)
FPS_vid = results['fps']
output_str = video + '---->' + '   FPS:   ' + str(FPS_vid) + '   op:   ' + str(OP_vid)
print(output_str)
