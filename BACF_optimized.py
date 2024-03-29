def BACF_optimized(params):
    """
    This function implements the background-aware correlation filter visual object tracker.
    :param params: dict: contains the following keys: 'video_path', 't_features', 't_global', 'search_area_shape',
    'search_area_scale', 'filter_max_area', 'learning_rate', 'output_sigma_factor', 'interpolate_response',
    'newton_iterations', 'number_of_scales', 'scale_step', 'wsize', 'init_pos', 's_frames', 'no_fram', 'seq_st_frame',
    'seq_en_frame', 'admm_iterations', 'admm_lambda', 'visualization'
    :return: dict: results containing bounding box position in the key 'res' and the frame rate in the key 'fps'
    """
    import math as m
    import numpy as np
    import cv2
    import time
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    from resizeDFT2 import resizeDFT2
    from get_pixels import get_pixels
    from get_features import get_features
    from resp_newton import resp_newton
    from get_subwindow_no_window import get_subwindow_no_window

    # Setting parameters
    search_area_scale = params['search_area_scale']  # size of training/detection area proportional to the target size
    output_sigma_factor = params['output_sigma_factor']
    learning_rate = params['learning_rate']
    filter_max_area = params['filter_max_area']
    nScales = params['number_of_scales']  # number of scale resolutions to check
    scale_step = params['scale_step']
    interpolate_response = params['interpolate_response']

    features = params['t_features']
    video_path = params['video_path']
    s_frames = params['s_frames']
    pos = np.floor(params['init_pos'])  # initial centre-point (y by x) of target bounding box
    target_sz = np.floor(params['wsize'])  # initial height and height of target bounding box

    visualization = params['visualization']
    num_frames = params['no_fram']
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
            featureRatio = params['t_global']['cell_size']
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
    elif params['search_area_shape'] == 'square':
        sz = np.tile(m.sqrt(np.prod(base_target_sz * search_area_scale)), [1, 2])  # ignores target aspect ratio
    elif params['search_area_shape'] == 'fix_padding':
        sz = base_target_sz + m.sqrt(np.prod(base_target_sz * search_area_scale) +
                                     (base_target_sz[0] - base_target_sz[1]) / 4) - \
             sum(base_target_sz) / 2  # const padding
    else:
        raise ValueError('Unknown "search_area_shape". Must be "proportional", "square", or "fix_padding".')

    # set the size to exactly match the cell size
    sz = np.round(sz[0] / featureRatio) * featureRatio
    use_sz = np.floor(sz / featureRatio)

    # construct the label function- correlation output, 2D gaussian function, with a peak located upon the target
    # np.roll acts a circular shift operator. This is used to compute all possible patches in the entire frame
    output_sigma = m.sqrt(np.prod(np.floor(base_target_sz / featureRatio))) * output_sigma_factor
    rg = np.roll(np.arange(-1 * np.floor((use_sz[0] - 1) / 2), np.ceil((use_sz[0] - 1)/2) + 1),
                 int(-1 * np.floor((use_sz[0] - 1) / 2)))
    cg = np.roll(np.arange(-1 * np.floor((use_sz[1] - 1) / 2), np.ceil((use_sz[1] - 1) / 2) + 1),
                 int(-1 * np.floor((use_sz[1] - 1) / 2)))
    [rs, cs] = np.meshgrid(rg, cg)
    rs = rs.T
    cs = cs.T
    # y is the desired correlation response at each point within the size of the filter
    y = np.exp(-0.5 * ((np.power(rs, 2) + np.power(cs, 2)) / np.power(output_sigma, 2)))
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
    try:
        im = cv2.imread(video_path + '/img/' + s_frames[0])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    except:
        try:
            im = cv2.imread(s_frames[0])
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except:
            im = cv2.imread(video_path + '/' + s_frames[0])
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Check to see if it is a color image or grayscale
    if im.shape[2] == 3:
        if np.all(np.equal(im[:,:,0], im[:,:,1])):
            colorImage = False
        else:
            colorImage = True
    else:
        colorImage = False
    
    if im.shape[2] > 1 and colorImage is False:
        im = im[:,:,0]
        im = im.reshape([im.shape[0], im.shape[1], 1])

    # create scale factors to check for object at various scale resolutions
    if nScales > 0:
        scale_exp = np.arange(-1 * np.floor((nScales - 1) / 2), np.ceil((nScales - 1) / 2) + 1)
        scaleFactors = scale_step ** scale_exp
        min_scale_factor = scale_step ** np.ceil(m.log(max(np.divide(5, sz))) / m.log(scale_step))
        max_scale_factor = scale_step ** np.floor(m.log(min(np.divide([im.shape[0],
                                                                       im.shape[1]],
                                                                      base_target_sz))) / m.log(scale_step))

    if interpolate_response >= 3:
        # pre-computes the grid that is used for score optimization
        ky = np.roll(np.arange(-1 * np.floor((use_sz[0] - 1) / 2), np.ceil((use_sz[0] - 1) / 2) + 1),
                     int(-1 * np.floor((use_sz[0] - 1) / 2)))
        kx = np.roll(np.arange(-1 * np.floor((use_sz[1] - 1) / 2), np.ceil((use_sz[1] - 1) / 2) + 1),
                     int(-1 * np.floor((use_sz[1] - 1) / 2)))  # --> SAME AS MATLAB
        kx = kx.T
        newton_iterations = params['newton_iterations']

    # initialize the projection matrix (x,y,h,w)
    rect_position = np.zeros([num_frames, 4])

    # allocate memory for multi-scale tracking
    multires_pixel_template = np.zeros([int(sz[0]), int(sz[1]), im.shape[2], nScales], dtype=np.uint8)
    small_filter_sz = np.floor(base_target_sz / featureRatio)
    start_time = time.time()

    loop_frame = 0
    for frame in range(0, len(s_frames)):
        try:
            im = cv2.imread(video_path + '/img/' + s_frames[frame])
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except:
            try:
                im = cv2.imread(s_frames[frame])
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            except:
                im = cv2.imread(video_path + '/' + s_frames[frame])
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        if im.shape[2] > 1 and colorImage is False:
            im = im[:, :, 0]
            im = im.reshape([im.shape[0], im.shape[1], 1])

        # do not estimate translation and scaling on the first frame, since we are just initializing the tracker
        if frame > 0:

            # The filter is applied on multiple resolutions of the search area. This is used to determine any changes in
            # scale of the target object
            for scale_ind in range(0, nScales):
                multires_pixel_template[:, :, :, scale_ind] = \
                    get_pixels(im, pos, np.round(sz * currentScaleFactor * scaleFactors[scale_ind]), sz)

            feat_term2, _ = get_features(multires_pixel_template, features, global_feat_params, None)
            xtf = np.zeros(feat_term2.shape, dtype=complex)
            for p in range(0, feat_term2.shape[2]):
                for n in range(0, feat_term2.shape[3]):
                    xtf[:,:,p,n] = np.fft.fft2(np.multiply(feat_term2[:,:,p,n], cos_window[:, :]))
            responsef = np.sum(np.multiply(np.conj(g_f)[:, :, :, None], xtf), axis=2)

            # if we undersampled features, we want to interpolate the response to have the same size as the image patch
            if interpolate_response == 2:
                # use dynamic interp size
                interp_sz = np.floor(y.shape * featureRatio * currentScaleFactor)

            responsef_padded = resizeDFT2(responsef, interp_sz)

            # Get the response in the spatial domain
            response = np.zeros(responsef_padded.shape)
            for n in range(0, responsef.shape[2]):
                response[:,:,n] = np.real(np.fft.ifft2(responsef_padded[:,:,n]))  # MAY HAVE AN ISSUE HERE NOT BEING SYMMETRIC -- therefore added real --> SAME AS MATLAB :D :)

            # find maximum peak
            if interpolate_response == 3:
                raise ValueError('Invalid parameter value for "interpolate_response"')
            elif interpolate_response == 4:
                [disp_row, disp_col, sind] = resp_newton(response, responsef_padded, newton_iterations, ky, kx, use_sz)

            # Check if the target has completely gone off the frame
            if np.isnan(disp_row) or np.isnan(disp_col):
                break

            # calculate translation vector
            if interpolate_response == 0 or 3 or 4:
                translation_vec = np.round(np.array([disp_row, disp_col]) * featureRatio *
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

        # extract features and perform windowing
        feat_term, _ = get_features(pixels, features, global_feat_params, None)
        xf = np.zeros([feat_term.shape[1], feat_term.shape[1], feat_term.shape[2]], dtype=complex)
        for n in range(0, feat_term.shape[2]):
            xf[:, :, n] = np.fft.fft2(np.multiply(feat_term[:, :, n, 0], cos_window[:, :]))
        if frame == 0:
            model_xf = xf
        else:
            model_xf = ((1 - learning_rate) * model_xf) + (learning_rate * xf)

        g_f = np.zeros(xf.shape)
        g_f = g_f.astype('float32')
        h_f = g_f
        l_f = g_f

        # parameters from the original paper
        mu = 1
        betha = 10
        mumax = 10000
        i = 1

        T = np.prod(use_sz)
        S_xx = np.sum(np.multiply(model_xf.conj(), model_xf), axis=2)
        params['admm_iterations'] = 2

        while i <= params['admm_iterations']:
            # Solve for G
            B = S_xx + (T * mu)
            S_lx = np.sum(np.multiply(model_xf.conj(), l_f), axis=2)
            S_hx = np.sum(np.multiply(model_xf.conj(), h_f), axis=2)

            # equation (10) in original paper
            g_f = (((1 / (T * mu)) * np.multiply(yf[:, :, None], model_xf)) - ((1 / mu) * l_f) + h_f) - \
                  np.divide((((1 / (T * mu)) * np.multiply(model_xf, np.multiply(S_xx, yf)[:, :, None])) -
                             ((1 / mu) * np.multiply(model_xf, S_lx[:, :, None])) +
                             (np.multiply(model_xf, S_hx[:, :, None]))), B[:, :, None])

            # solve for H
            # Equation (6) in original paper
            h = np.zeros([g_f.shape[0], g_f.shape[1], g_f.shape[2]], dtype=complex)
            for n in range(0, g_f.shape[2]):
                h[:, :, n] = (T / ((mu * T) + params['admm_lambda'])) * np.fft.ifft2((mu * g_f[:, :, n]) + l_f[:, :, n])

            [sx, sy, h] = get_subwindow_no_window(h, np.floor(use_sz / 2), small_filter_sz)
            t = np.zeros([int(use_sz[0]), int(use_sz[1]), h.shape[2]], dtype=complex)
            t[int(sx[0]):int(sx[-1])+1, int(sy[0]):int(sy[-1])+1, :] = h

            h_f = np.zeros([t.shape[1], t.shape[1], t.shape[2]], dtype=complex)
            for n in range(0, t.shape[2]):
                h_f[:, :, n] = np.fft.fft2(t[:, :, n])

            # update L
            l_f = l_f + (mu * (g_f - h_f))

            # update mu- betha = 10
            mu = min(betha * mu, mumax)
            i = i + 1

        target_sz = np.floor(base_target_sz * currentScaleFactor)

        # save position and calculate FPS
        rect_position[loop_frame, :] = np.concatenate((pos[1::-1] - np.floor(target_sz[1::-1] / 2), target_sz[1::-1]))

        elapsed = time.time() - start_time

        # visualization
        if visualization == 1:
            rect_position_vis = np.concatenate((pos[1::-1] - (target_sz[1::-1] / 2), target_sz[1::-1]))
            im_to_show = im / 255
            if im_to_show.shape[2] == 1:
                im_to_show = np.tile(im_to_show, [1, 1, 3])  # if grayscale, ensure it plots image in grayscale
            if frame == 0:
                fig, ax = plt.subplots(1, num='Tracking')
                ax.imshow(im_to_show)
                rect = patches.Rectangle(rect_position_vis[0:2], rect_position_vis[2], rect_position_vis[3],
                                     linewidth=3, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
                ax.annotate(str(frame + 1), [10, 10], color='c')
                ax.axis('off')  # remove axis values
                ax.axis('image')  # scale image appropriately
                ax.set_position([0, 0, 1, 1])
                fig.show()
                fig.canvas.draw()
            else:
                ax.clear()
                resp_sz = np.round(sz * currentScaleFactor * scaleFactors[scale_ind])
                xs = np.floor(old_pos[1]) + (np.arange(1, resp_sz[1] + 1)) - np.floor(resp_sz[1] / 2)
                ys = np.floor(old_pos[0]) + (np.arange(1, resp_sz[0] + 1)) - np.floor(resp_sz[0] / 2)
                sc_ind = np.floor((nScales - 1) / 2)

                ax.imshow(im_to_show)
                ax.axis('off')  # remove axis values
                ax.axis('image')  # scale image appropriately
                ax.set_position([0, 0, 1, 1])
                im_overlay = np.fft.fftshift(response[:,:,int(sc_ind)])
                ax.imshow(im_overlay, extent=[xs[0], xs[-1], ys[0], ys[-1]],
                          alpha=0.2, cmap='hsv')
                rect.remove()  # remove previous rectangle each time
                rect = patches.Rectangle(rect_position_vis[0:2], rect_position_vis[2], rect_position_vis[3],
                                         linewidth=3, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
                frame_str = '# Frame: ' + str(loop_frame + 1) + ' / ' + str(num_frames)
                FPS_str = 'FPS: ' + str(1 / (elapsed / loop_frame))
                ax.annotate(frame_str, [20, 30], color='r', backgroundcolor='w')
                ax.annotate(FPS_str, [20, 60], color='r', backgroundcolor='w', fontsize=16)
                fig.show()
                fig.canvas.draw()  # used to update the plot in real-time

        loop_frame += 1

    # Save results
    fps = loop_frame / elapsed
    results = {'res': rect_position, 'fps': fps}

    return results
