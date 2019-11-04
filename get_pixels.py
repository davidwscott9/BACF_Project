def get_pixels(im, pos, sz, resize_target):

    import numpy as np
    import cv2
    xs = np.floor(pos[1]) + np.arange(0, sz[1]) - np.floor(sz[1] / 2)
    ys = np.floor(pos[0]) + np.arange(0, sz[0]) - np.floor(sz[0] / 2)

    # check for out-of-bounds coordinates, and set them to the values at the borders
    xs[xs < 1] = 1
    ys[ys < 1] = 1
    xs[xs > im.shape[1]] = im.shape[1]
    ys[ys > im.shape[0]] = im.shape[0]

    # extract image
    im_patch = im[min(ys):max(ys)+1, min(xs):max(xs)+1, :]
    resized_patch = cv2.resize(im_patch, (resize_target[0], resize_target[1]))

    return resized_patch