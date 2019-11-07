def run_BACF(seq, video_path, lr):

    import math as m
    import numpy as np
    from BACF_optimized import BACF_optimized
    # Default parameters used in the ICCV 2017 BACF paper

    # HOG feature parameters
    hog_params = {'nDim': 9}  # 9 if HOG, 31 if FHOG
    # Grayscale feature parameters (DON"T THINK THESE ARE EVER USED)
    colorspace = 'gray'
    nDim = 1
    grayscale_params = {'colorspace': colorspace, 'nDim': nDim}

    # Global feature parameters
    t_features = {'fparams': hog_params}  # Omitted the 'GetFeatures' key. Just using FHog when appropriate.
    cell_size = 4  # feature cell size
    cell_selection_thresh = 0.75**2  # threshold for reducing the cell size in low-resolution cases
    t_global = {'cell_size': cell_size, 'cell_selection_thresh': cell_selection_thresh}

    # Search region + extended background parameters
    search_area_shape = 'square'  # shape of the training/detection window: 'proportional', 'square', or 'fix_padding'
    search_area_scale = 5  # the size of the training/detection area proportional to the target size
    filter_max_area = 50**2  # the size of the training/detection area in feature grid cells

    # Learning Parameters
    learning_rate = lr  # learning rate
    output_sigma_factor = 1/16  # standard deviation of the desired correlation output (proportional to target)

    # Detection parameters
    # correlation score interpolation strategy: 0 - off, 1 - feature grid, 2 - pixel grid, 4 - Newton's method
    interpolate_response = 4
    newton_iterations = 50  # number of Newton's iteration to maximize the detection scores

    # Scale parameters
    number_of_scales = 5
    scale_step = 1.01

    # size, position, frames initialization
    wsize = np.array([seq['init_rect'][3], seq['init_rect'][2]])
    init_pos = np.array([seq['init_rect'][1], seq['init_rect'][0]]) + np.floor(wsize / 2)
    s_frames = seq['s_frames']
    no_fram = seq['en_frame'] - seq['st_frame'] + 1
    seq_st_frame = seq['st_frame']
    seq_en_frame = seq['en_frame']

    # ADMM parameters, number of iterations, and lambda-mu and beta are set in the main function.
    admm_iterations = 2
    admm_lambda = 0.01

    # Debug and visualization
    visualization = 1

    # combine variables into param dictionary
    params = {'video_path': video_path, 't_features': t_features, 't_global': t_global,
              'search_area_shape': search_area_shape, 'search_area_scale': search_area_scale,
              'filter_max_area': filter_max_area, 'learning_rate': learning_rate,
              'output_sigma_factor': output_sigma_factor, 'interpolate_response': interpolate_response,
              'newton_iterations': newton_iterations, 'number_of_scales': number_of_scales,
              'scale_step': scale_step, 'wsize': wsize, 'init_pos': init_pos, 's_frames': s_frames,
              'no_fram': no_fram, 'seq_st_frame': seq_st_frame, 'seq_en_frame': seq_en_frame,
              'admm_iterations': admm_iterations, 'admm_lambda': admm_lambda, 'visualization': visualization}

    # run the main function
    results = BACF_optimized(params)
    return results
