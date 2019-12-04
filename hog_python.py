def hog_python(I, binSize, nOrients):
    """
    This function computes the HOG descriptors for a given image
    :param I: The input image
    :param binSize: int: size of the pixel bin
    :param nOrients: int: number of orientations. Standard is 9
    :return: numpy array: HOG features
    """
    import cv2
    binSize = int(binSize)
    block_size = 1
    hog = cv2.HOGDescriptor(_winSize=(I.shape[1] // binSize * binSize,
                                  I.shape[0] // binSize * binSize),
                        _blockSize=(block_size * binSize,
                                    block_size * binSize),
                        _blockStride=(binSize, binSize),
                        _cellSize=(binSize, binSize),
                        _nbins=nOrients)

    n_cells = (I.shape[0] // binSize, I.shape[1] // binSize)
    H = hog.compute(I)\
               .reshape(n_cells[1] - block_size + 1,
                        n_cells[0] - block_size + 1,
                        block_size, block_size, nOrients) \
               .transpose((1, 0, 2, 3, 4))

    H = H.reshape(n_cells[1] - block_size + 1,
                        n_cells[0] - block_size + 1, -1)

    # This function can only do 1 image at a time

    return H
