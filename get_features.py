def get_features(image, features, gparams, fg_size):
    """
    Obtains the HOG features for a given image or image patch
    :param image: A grayscale or color image
    :param features: dict detailing the feature parameters
    :param gparams: dict detailing HOG parameters
    :param fg_size: List of length=2 detailing the size of the figure
    :return:
    numpy array: 4D array detailing the features of the image
    list: list of the form [height, width] detailing the image image size
    """
    import numpy as np
    from get_hog import get_hog

    if len(image.shape) == 2:
        image = image.reshape([image.shape[0], image.shape[1], 1])

    if len(image.shape) == 3:
        [im_height, im_width, num_im_chan] = image.shape
        num_images = 1
    else:
        [im_height, im_width, num_im_chan, num_images] = image.shape

    tot_feature_dim = features['fparams']['nDim']

    if fg_size is None or (not fg_size is True):
        if gparams['cell_size'] == -1:
            fg = get_hog(image, features['fparams'], gparams)
            fg_size = fg.shape
        else:
            fg_size = [np.floor(im_height / gparams['cell_size']), np.floor(im_width / gparams['cell_size'])]

    feature_image = get_hog(image, features['fparams'], gparams)
    if num_images == 1:
        feature_image = feature_image.reshape(feature_image.shape[0], feature_image.shape[1],
                                              feature_image.shape[2], 1)

    feature_pixels = np.zeros([int(fg_size[0]), int(fg_size[1]), tot_feature_dim, num_images])
    feature_pixels[:, :, 0::, :] = feature_image
    support_sz = [im_height, im_width]

    return feature_pixels, support_sz
