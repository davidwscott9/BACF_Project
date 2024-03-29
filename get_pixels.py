def get_pixels(im, pos, sz, resize_target):
    """
    This function gets an image patch given an original image, position, size, and resize target
    :param im: Original, full-size image
    :param pos: The position of the patch
    :param sz: list: 1x2 detailing the size of the image
    :param resize_target: list: 1x2 detailing the target size of the image patch
    :return: image patch
    """
    import numpy as np
    import cv2
    xs = np.floor(pos[1]) + np.arange(0, sz[1]) - np.floor(sz[1] / 2)
    ys = np.floor(pos[0]) + np.arange(0, sz[0]) - np.floor(sz[0] / 2)

    # check for out-of-bounds coordinates, and set them to the values at the borders
    xs[xs < 1] = 0
    ys[ys < 1] = 0
    xs[xs >= im.shape[1]] = im.shape[1] - 1
    ys[ys >= im.shape[0]] = im.shape[0] - 1

    # extract image
    im_patch = im[int(min(ys)):int(max(ys))+1, int(min(xs)):int(max(xs)+1), :]
    resized_patch = cv2.resize(im_patch, (int(resize_target[0]), int(resize_target[1])))

    if len(resized_patch.shape) == 2:
        resized_patch = resized_patch.reshape([resized_patch.shape[0], resized_patch.shape[1], 1])
    return resized_patch