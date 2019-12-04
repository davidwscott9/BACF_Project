def get_subwindow_no_window(im, pos, sz):
    """
    Obtains a subwindow from an image with replication padding. Adapted from João F. Henriques, 2012
    http://www.isr.uc.pt/~henriques/
    :param im: the original image
    :param pos: list with the form [x,y] - the centre of the subwindow
    :param sz: list of the form [height, width] - the size of the subwindow
    :return: subwindow x-dimensions (list), subwindow y-dimensions (list), subwindow (numpy array)
    """
    import numpy as np
    if len(sz) == 1:
        sz = np.array([sz, sz])

    xs = np.floor(pos[1]) + np.arange(0, sz[1]) - np.floor(sz[1] / 2)
    xxs = xs
    ys = np.floor(pos[0]) + np.arange(0, sz[0]) - np.floor(sz[0] / 2)
    yys = ys

    # check for out-of-bounds coordinates, and set them to the values at the borders
    xs[xs < 1] = 1
    ys[ys < 1] = 1
    xs[xs > im.shape[1]] = im.shape[1]
    ys[ys > im.shape[0]] = im.shape[0]

    # extract image
    out = im[int(min(ys)):int(max(ys))+1, int(min(xs)):int(max(xs))+1, :]

    return yys, xxs, out