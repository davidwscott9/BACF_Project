def get_features(image, features, gparams, fg_size):

    import numpy as np
    from get_fhog import get_fhog
    # IGNORING IF STATEMENT. ASSUMING IT JUST IS A PYTHON DICT

    if len(image.shape) == 2:
        image = image.reshape([image.shape[0], image.shape[1], 1])

    if len(image.shape) == 3:
        [im_height, im_width, num_im_chan] = image.shape
        num_images = 1
    else:
        [im_height, im_width, num_im_chan, num_images] = image.shape
    colorImage = num_im_chan == 3

    # compute the total dimension of all features
    ## tot_feature_dim = 0

    # ANOTHER USELESS LOOKING FOR LOOP
    tot_feature_dim = features['fparams']['nDim']

    if fg_size is None or (not fg_size is True):
        if gparams['cell_size'] == -1:
            fg = get_fhog(image, features['fparams'], gparams)
            fg_size = fg.shape
        else:
            fg_size = [np.floor(im_height / gparams['cell_size']), np.floor(im_width / gparams['cell_size'])]

    # IGNORING CELL_SIZE < 0 CASE BECAUSE WHEN WOULD THAT HAPPEN???
    feature_image = get_fhog(image, features['fparams'], gparams)
    if num_images == 1:
        feature_image = feature_image.reshape(feature_image.shape[0], feature_image.shape[1],
                                              feature_image.shape[2], 1)

    feature_pixels = np.zeros([int(fg_size[0]), int(fg_size[1]), tot_feature_dim, num_images])
    feature_pixels[:, :, 0::, :] = feature_image
    support_sz = [im_height, im_width]

    return feature_pixels, support_sz
