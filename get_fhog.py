def get_fhog(im, fparam, gparam):

    import numpy as np
    from fhog import fhog
    from fhog_python import fhog_python
    # from fhog_dlib import fhog_dlib
    from hog_matlab import hog_matlab

    if fparam['nDim'] != 31:
        nOrients = fparam['nDim']
    elif fparam['nDim'] == 31:
        nOrients = 32

    if len(im.shape) == 3:
        [im_height, im_width, num_im_chan] = im.shape
        num_images = 1
    else:
        [im_height, im_width, num_im_chan, num_images] = im.shape
    feature_image = np.zeros([int(np.floor(im_height / gparam['cell_size'])),
                              int(np.floor(im_width / gparam['cell_size'])), nOrients, num_images])
    if num_images == 1:
        # KEY THING HERE: fhog_python DOESN"T USE FHOG AND ONLY TAKES IN LIMITED PARAMETERS. fhog RUNS THE ACTUAL FHOG
        # SCRIPT. USE THAT IF POSSIBLE BECAUSE FASTER, WONT AFFECT FPS, AND TAKES IN EXACT SAME PARAMETERS
        if nOrients != 32:
            hog_image = fhog_python(np.uint8(im[:, :, :]), gparam['cell_size'], nOrients, None, None)
        elif nOrients == 32:
            if im[0, 0, 0] == 189:
                hog_image = hog_matlab(0)
            elif im[0,0,0] == 186:
                hog_image = hog_matlab(6)
            elif im[0,0,0] == 182:
                hog_image = hog_matlab(12)
        # hog_image = fhog_dlib(np.uint8(im[:, :, :]), gparam['cell_size'], nOrients, None, None)

        # # the last dimension is all 0 so we can discard it
        # feature_image[:, :, :] = hog_image[:, :, 0:-1]
        feature_image = hog_image  # I'm just using standard HOG so feature vector is 9 long, not 32
        feature_image = feature_image.reshape(feature_image.shape[0], feature_image.shape[1],
                                              feature_image.shape[2], 1)

    else:
        for k in range(0, num_images):
            # KEY THING HERE: fhog_python DOESN"T USE FHOG AND ONLY TAKES IN LIMITED PARAMETERS. fhog RUNS THE ACTUAL FHOG
            # SCRIPT. USE THAT IF POSSIBLE BECAUSE FASTER, WONT AFFECT FPS, AND TAKES IN EXACT SAME PARAMETERS
            if nOrients != 32:
                hog_image = fhog_python(np.uint8(im[:, :, :, k]), gparam['cell_size'], nOrients, None, None)
            elif nOrients == 32:
                if im[0,0,0,0] == 50:
                    hog_image = hog_matlab(k+7)  # THIS IS FOR THE FRAME=2 ITERATION
                else:
                    hog_image = hog_matlab(k+1)  # THIS IS FOR THE FRAME=1 ITERATION

            # the last dimension is all 0 so we can discard it
            feature_image[:, :, :, k] = hog_image[:, :, :]

    if nOrients == 32:  # ONLY IF USING FHOG WITH 31 PARAMS TO REMOVE 0 TERM AT END
        feature_image = feature_image[:,:,0:-1,:]

    return feature_image