def get_fhog(im, fparam, gparam):

    import numpy as np
    nOrients = 9

    [im_height, im_width, num_in_chan, num_images] = im.shape
    feature_image = np.zeros([np.floor(im_height, gparam['cell_size']), np.floor(im_width / gparam['cell_size']),
                             fparam['nDim'], num_images])

    for k in range(0, num_images):
        hog_image = fhog((im[:, :, k]).astype('float32'), gparam['cell_size'], nOrients)

        # the last dimension is all 0 so we can discard it
        feature_image[:, :, :, k] = hog_image[:, :, 0:-1]

    return feature_image