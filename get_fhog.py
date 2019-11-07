def get_fhog(im, fparam, gparam):

    import numpy as np
    from fhog import fhog
    from fhog_python import fhog_python
    nOrients = 9

    if len(im.shape) == 3:
        [im_height, im_width, num_im_chan] = im.shape
        num_images = 1
    else:
        [im_height, im_width, num_im_chan, num_images] = im.shape
    feature_image = np.zeros([int(np.floor(im_height / gparam['cell_size'])),
                              int(np.floor(im_width / gparam['cell_size'])), fparam['nDim'], num_images])

    if num_images == 1:
        # KEY THING HERE: fhog_python DOESN"T USE FHOG AND ONLY TAKES IN LIMITED PARAMETERS. fhog RUNS THE ACTUAL FHOG
        # SCRIPT. USE THAT IF POSSIBLE BECAUSE FASTER, WONT AFFECT FPS, AND TAKES IN EXACT SAME PARAMETERS
        hog_image = fhog_python(np.uint8(im[:, :, :]), gparam['cell_size'], nOrients, None, None)

        # # the last dimension is all 0 so we can discard it
        # feature_image[:, :, :] = hog_image[:, :, 0:-1]
        feature_image = hog_image # I'm just using standard HOG so feature vector is 9 long, not 32

    else:
        for k in range(0, num_images):
            # KEY THING HERE: fhog_python DOESN"T USE FHOG AND ONLY TAKES IN LIMITED PARAMETERS. fhog RUNS THE ACTUAL FHOG
            # SCRIPT. USE THAT IF POSSIBLE BECAUSE FASTER, WONT AFFECT FPS, AND TAKES IN EXACT SAME PARAMETERS
            hog_image = fhog_python(np.uint8(im[:, :, k]), gparam['cell_size'], nOrients, None, None)

            # the last dimension is all 0 so we can discard it
            feature_image[:, :, :, k] = hog_image[:, :, 0:-1]


    return feature_image