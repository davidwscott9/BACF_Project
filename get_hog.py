def get_hog(im, fparam, gparam):
    """
    This function acquires and formats HOG features for a given input image. Accepts either a single image or multiple
    :param im: The original input image, either grayscale or color in the form [im_height, im_width, num_im_chan] if a
    single image or [im_height, im_width, num_im_chan, num_images] if multiple images
    :param fparam: dict: contains the number of orientations for the HOG descriptor
    :param gparam: dict: contains the cell (bin) size for the HOG descriptor
    :return: HOG features of the input image, im
    """
    import numpy as np
    from hog_python import hog_python

    nOrients = fparam['nDim']

    if len(im.shape) == 2:
        im = im.reshape([im.shape[0], im.shape[1], 1])
    if len(im.shape) == 3:
        [im_height, im_width, num_im_chan] = im.shape
        num_images = 1
    else:
        [im_height, im_width, num_im_chan, num_images] = im.shape
    feature_image = np.zeros([int(np.floor(im_height / gparam['cell_size'])),
                              int(np.floor(im_width / gparam['cell_size'])), nOrients, num_images])
    if num_images == 1:
        hog_image = hog_python(np.uint8(im[:, :, :]), gparam['cell_size'], nOrients)

        feature_image = hog_image  # I'm just using standard HOG so feature vector is 9 long, not 32
        feature_image = feature_image.reshape(feature_image.shape[0], feature_image.shape[1],
                                              feature_image.shape[2], 1)

    else:
        for k in range(0, num_images):
            hog_image = hog_python(np.uint8(im[:, :, :, k]), gparam['cell_size'], nOrients)
            feature_image[:, :, :, k] = hog_image[:, :, :]

    return feature_image
