def BACF_optimized(params):

    import math as m
    import numpy as np
    import cv2
    import time
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    # Setting parameters for local use.
    search_area_scale = params['search_area_scale']
    output_sigma_factor = params['output_sigma_factor']
    learning_rate = params['learning_rate']
    filter_max_area = params['filter_max_area']
    nScales = params['number_of_scales']
    scale_step = params['scale_step']
    interpolate_response = params['interpolate_response']

    features = params['t_features']
    video_path = params['video_path']
    s_frames = params['s_frames']
    pos = np.floor(params['init_pos'])
    target_sz = np.floor(params['wsize'])

    visualization = params['visualization']
    num_frames = params['no_frame']
    init_target_sz = target_sz

    # Set the features ratio to the feature-cell size
    featureRatio = params['t_global']['cell_size']
    search_area = np.prod(init_target_sz / featureRatio * search_area_scale)

    # when the number of cells are small, choose a smaller cell size
    if 'cell_selection_thresh' in params['t_global']:
        if search_area < params['t_global']['cell_selection_thresh'] * filter_max_area:
            params['t_global']['cell_size'] = min(featureRatio,
                                                  max(1, np.ceil(m.sqrt(np.prod(init_target_sz * search_area_scale) /
                                                                       (params['t_global']['cell_selection_thresh'] *
                                                                        filter_max_area)))))
            featureRatio = params['t_global_cell']['cell_size']
            search_area = np.prod(init_target_sz / featureRatio * search_area_scale)

    global_feat_params = params['t_global']

    if search_area > filter_max_area:
        currentScaleFactor = m.sqrt(search_area / filter_max_area)
    else:
        currentScaleFactor = 1.0

    # target size at the initial scale
    base_target_sz = target_sz / currentScaleFactor

    # window size, taking padding into account
    if params['search_area_shape'] == 'proportional':
        sz = np.floor(base_target_sz * search_area_scale)  # proportional area, same aspect ratio as the target
    elif params['search_area_shape'] == ' square':
        sz = np.tile(m.sqrt(np.prod(base_target_sz * search_area_scale)), [1, 2])  # ignores target aspect ratio
    elif params['search_area_shape'] == 'fix_padding':
        sz = base_target_sz + m.sqrt(np.prod(base_target_sz * search_area_scale) +
                                     (base_target_sz[0] - base_target_sz[1]) / 4) - \
             sum(base_target_sz) / 2  # const padding
    else:
        raise ValueError('Unknown "search_area_shape". Must be "proportional", "square", or "fix_padding".')

    # set the size to exactly match the cell size
    sz = np.round(sz / featureRatio) * featureRatio
    use_sz = np.floor(sz / featureRatio)

    # construct the label function- correlation output, 2D gaussian function, with a peak located upon the target
    output_sigma = m.sqrt(np.prod(np.floor(base_target_sz / featureRatio))) * output_sigma_factor
    rg = np.roll(np.arange(-1 * np.floor((use_sz[0] - 1) / 2), np.ceil((use_sz[0] - 1)/2) + 1),
                 -1 * np.floor((use_sz[0] - 1) / 2), axis=1)  # THIS MAY BE A SOURCE OF AN ERROR LOTS OF CONVERTING HERE
    cg = np.roll(np.arange(-1 * np.floor((use_sz[1] - 1) / 2), np.ceil((use_sz[1] - 1) / 2) + 1),
                 -1 * np.floor((use_sz[1] - 1) / 2), axis=1)  # THIS MAY BE A SOURCE OF AN ERROR LOTS OF CONVERTING HERE
    [rs, cs] = np.meshgrid(rg, cg)  # MAY BE ANOTHER CANDIDATE ERROR SOURCE
    rs = rs.T
    cs = cs.T
    # MAY NEED TO CONVERT VARIABLES TO NP ARRAYS IF GETTING ERRORS BECAUSE I'M USING NP.POWER, etc.
    y = m.exp(-0.5 * ((np.power(rs, 2) + np.power(cs, 2)) / np.power(output_sigma, 2)))
    yf = np.fft.fft2(y)  # fast fourier transform of y

    if interpolate_response == 1:
        interp_sz = use_sz * featureRatio
    else:
        interp_sz = use_sz

    # construct cosine window
    term1 = np.array([np.hanning(use_sz[0])])
    term2 = np.array([np.hanning(use_sz[1])])
    cos_window = np.matmul(term1.T, term2)
    cos_window = cos_window.astype('float32')

    # Calculate feature dimension
    # IF SOMETHING IS WRONG HERE, MIGHT BE ABLE TO CIRCUMVENT THE TRY/EXCEPT BY ENSURING IMAGE FILES ARE WELL FORMATTED
    try:
        im = cv2.imread(video_path + '/img/' + s_frames[0])
    except:
        try:
            im = cv2.imread(s_frames[0])
        except:
            im = cv2.imread(video_path + '/' + s_frames[0])

    if im.shape[2] == 3:
        if np.all(np.equal(im[:,:,0], im[:,:,1])):
            colorImage = False
        else:
            colorImage = True
    else:
        colorImage = False

    # Compute feature dimensionality
    # MOST EVERYTHING IN THIS SECTION SEEMS TO HAVE NO VALUE
    feature_dim = features['fparams']['nDim']
    
    if im.shape[2] > 1 and colorImage is False:
        im = im[:,:,0]
    
    if nScales > 0:
        scale_exp = np.arange(-1 * np.floor((nScales - 1) / 2), np.ceil((nScales - 1) / 2) + 1)
        scaleFactors = scale_step ** scale_exp
        min_scale_factor = scale_step ^ np.ceil(m.log(max(np.divide(5, sz))) / m.log(scale_step))
        max_scale_factor = scale_step ^ np.floor(m.log(min([im.shape[0]/base_target_sz,
                                                            im.shape[1]/base_target_sz])) / m.log(scale_step))

    if interpolate_response >= 3:
        # pre-computes the grid that is used for score optimization
        ky = np.roll(np.arange(-1 * np.floor((use_sz[0] - 1) / 2), np.ceil((use_sz[0] - 1) / 2) + 1),
                     [1, -1 * np.floor((use_sz[0] - 1) / 2)], axis=[0, 1])
        kx = np.roll(np.arange(-1 * np.floor((use_sz[1] - 1) / 2), np.ceil((use_sz[1] - 1) / 2) + 1),
                     [1, -1 * np.floor((use_sz[1] - 1) / 2)], axis=[0, 1])
        kx = kx.T  # MAKE SURE KX ACTUALLY AN NDARRAY AND NOT JUST 1D OTHERWISE TRANSPOSE WON"T WORK!
        newton_iterations = params['newton_iterations']

    # initialize the projection matrix (x,y,h,w)
    rect_position = np.zeros([num_frames, 4])
    ## time = 0

    # allocate memory for multi-scale tracking
    multires_pixel_template = np.zeros([sz[0], sz[1], im.shape[2], nScales], dtype=np.uint8)
    small_filter_sz = np.floor(base_target_sz / featureRatio)

    loop_frame = 0
    for frame in range(0, s_frames.size):  # S_FRAMES NEEDS TO BE A NP ARRAY FOR .SIZE TO WORK
        try:
            im = cv2.imread(video_path + '/img/' + s_frames[frame])
        except:
            try:
                im = cv2.imread(s_frames[frame])
            except:
                im = cv2.imread(video_path + '/' + s_frames[frame])

        if im.shape[2] > 1 and colorImage is False:
            im = im[:,:,0]

        t = time.time()   # LATER USE elapsed = time.time() - t
        elapsed = 0

        # do not estimate translation and scaling on the first frame, since we just want to initialize the tracker there
        if frame > 0:
            for scale_ind in range(0, nScales):
                multires_pixel_template[:, :, :, scale_ind] = \
                    get_pixels(im, pos, np.round(sz * currentScaleFactor * scaleFactors[scale_ind]), sz)

            xtf = np.fft.fft2(np.multiply(get_features(multires_pixel_template,
                                                       features, global_feat_params), cos_window))
            mat1 = np.sum(np.multiply(np.conj(g_f), xtf), axis=2)
            responsef = np.reshape(mat1, [mat1.shape[0], mat1.shape[1], mat1.shape[3], mat1.shape[2]])

            # if we undersampled features, we want to interpolate the response to have the same size as the image patch
            if interpolate_response == 2:
                # use dynamic interp size
                interp_sz = np.floor(y.shape * featureRatio * currentScaleFactor)

            responsef_padded = resizeDFT2(responsef, interp_sz)

            # response in the spatial domain
            response = np.fft.ifft2(responsef_padded) # MAY HAVE AN ISSUE HERE NOT BEING SYMMETRIC

            # find maximum peak
            if interpolate_response == 3:
                raise ValueError('Invalid parameter value for "interpolate_response"')
            elif interpolate_response == 4: # IF GETTING AN ERROR WITH SIND, MAY BE BECAUSE IT NEEDS TO INDEX AT SIND-1
                [disp_row, disp_col, sind] = resp_newton(response, responsef_padded, newton_iterations, ky, kx, use_sz)
            # MAY NOT NEED THIS  SECTION. TOO MANY CONVERSIONS. SKIPPING FOR NOW. SEEMS LIKE interpolate_response = 4 always
            #else:
               # [row, col, sind] = np.unravel_index(response.shape, find)

            # calculate translation
            if interpolate_response == 0 or 3 or 4:
                translation_vec = np.round([disp_row, disp_col] * featureRatio *
                                           currentScaleFactor * scaleFactors[sind])
            elif interpolate_response == 1:
                translation_vec = np.round([disp_row, disp_col] * currentScaleFactor * scaleFactors[sind])
            elif interpolate_response == 2:
                translation_vec = np.round([disp_row, disp_col] * scaleFactors[sind])

            # set the scale
            currentScaleFactor = currentScaleFactor * scaleFactors[sind]

            # adjust to make sure we are not too large or too small
            if currentScaleFactor < min_scale_factor:
                currentScaleFactor = min_scale_factor
            elif currentScaleFactor > max_scale_factor:
                currentScaleFactor = max_scale_factor

            # update position
            old_pos = pos
            pos = pos + translation_vec

        # extract training sample image region
        pixels = get_pixels(im, pos, np.round(sz*currentScaleFactor), sz)

        # extract features and do windowing
        xf = np.fft.fft2(np.multiply(get_features(pixels, features, global_feat_params), cos_window))

        if frame == 0:
            model_xf = xf
        else:
            model_xf = ((1 - learning_rate) * model_xf) + (learning_rate * xf)

        g_f = np.zeros(xf.shape)
        g_f = g_f.astype('float32')
        h_f = g_f
        l_f = g_f
        mu = 1
        betha = 10
        mumax = 10000
        i = 1

        T = np.prod(use_sz)
        S_xx = np.sum(np.multiply(model_xf.conj(), model_xf, axis=2))
        params['admm_iterations'] = 2

        while i <= params['admm_iterations']:
            # Solve for G
            B = S_xx + (T * mu)
            S_lx = np.sum(np.multiply(model_xf.conj(), l_f, axis=2))
            S_hx = np.sum(np.multiply(model_xf.conj(), h_f, axis=2))
            g_f = (((1 / (T * mu)) * np.multiply(yf, model_xf)) - ((1 / mu) * l_f) + h_f) - \
                  np.divide((((1 / (T * mu)) * np.multiply(model_xf, np.multiply(S_xx, yf))) -
                             ((1 / mu) * np.multiply(model_xf, S_lx)) + (np.multiply(model_xf, S_hx))), B)

            # solve for H
            h = (T / ((mu * T) + params['admm_lambda'])) * np.fft.ifft2((mu * g_f) + l_f)
            [sx, sy, h] = get_subwindow_no_window(h, np.floor(use_sz / 2), small_filter_sz)
            t = np.zeros([use_sz[0], use_sz[1], h.shape[2]])
            t = t.astype('float32')
            t[sx, sy, :] = h # SX AND SY SHOULD BE VALUE-1 FOR PROPER INDEXING
            h_f = np.fft.fft2(t)

            # update L
            l_f = l_f + (mu * (g_f - h_f))

            # update mu- betha = 10
            mu = min(betha * mu, mumax)
            i = i + 1

        target_sz = np.floor(base_target_sz * currentScaleFactor)

        # save position and calculate FPS
        rect_position[loop_frame, :] = np.concatenate((pos[1::-1] - np.floor(target_sz[1::-1] / 2), target_sz[1::-1]))  # should be 1x4

        elapsed = elapsed + (time.time() - t)

        # visualization
        if visualization == 1:
            rect_position_vis = np.concatenate((pos[1::-1] - (target_sz[1::-1] / 2), target_sz[1::-1]))  # should be 1x4
            im_to_show = im / 255
            if im_to_show.shape[2] == 1:
                np.tile(im_to_show, [1, 1, 3])
            if frame == 1:
                fig, ax = plt.subplots(1, num='Tracking')
                ax.imshow(im_to_show)  # MATLAB CODE HAS IMAGESC INSTEAD OF IMSHOW. HOPEFULLY NOT A BIG DIFFERENCE
                rect = patches.Rectangle(rect_position_vis[0:2], rect_position_vis[2], rect_position_vis[3],
                                     linewidth=2, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
                ax.annotate(str(frame + 1), [10, 10], color='c')
                ax.axis('off')  # remove axis values
                ax.axis('image')  # scale image appropriately
                ax.set_position([0, 0, 1, 1])
            else:
                resp_sz = np.round(sz * currentScaleFactor * scaleFactors[scale_ind])
                xs = np.floor(old_pos[1]) + (np.arange(1, resp_sz[1] + 1)) - np.floor(resp_sz[1] / 2)
                ys = np.floor(old_pos[0]) + (np.arange(1, resp_sz[0] + 1)) - np.floor(resp_sz[0] / 2)
                sc_ind = np.floor((nScales - 1) / 2)  # NOT INCLUDING PLUS 1 SINCE IT'S AN INDEX

                ax.imshow(im_to_show)
                im_overlay = np.fft.fftshift(response[:,:,sc_ind])
                ax.imshow(im_overlay, extent=[xs, xs + im_overlay.shape[1], ys, ys + im_overlay[0]],
                          alpha=0.2, cmap='hsv')
                rect = patches.Rectangle(rect_position_vis[0:2], rect_position_vis[2], rect_position_vis[3],
                                         linewidth=2, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
                frame_str = '# Frame: ' + str(loop_frame + 1) + ' / ' + str(num_frames)
                FPS_str = 'FPS: ' + str(1 / (elapsed / loop_frame))
                ax.annotate(frame_str, [20, 30], color='r', backgroundcolor='w')
                ax.annotate(FPS_str, [20, 60], color='r', backgroundcolor='w', fontsize=16)

        loop_frame += 1

    # Save results
    fps = loop_frame / elapsed
    results = {'type': 'rect', 'res': rect_position, 'fps': fps}

    return results
