def resizeDFT2(inputdft, desiredSize):
    """
    Resizes a discrete fourier transform (dft) input to a desired size
    :param inputdft: numpy array of original dft
    :param desiredSize: list of the form [height, width]
    :return: numpy array of the resized dft
    """
    import numpy as np
    [imh, imw, n1] = inputdft.shape
    n2 = 1
    imsz = [imh, imw]

    if desiredSize[0] != imsz[0] or desiredSize[1] != imsz[1]:
        minsz = np.minimum(imsz, desiredSize)

        scaling = np.prod(desiredSize) / np.prod(imsz)

        resizeddft = np.zeros([desiredSize, n1, n2], dtype=np.complex_)

        mids = np.ceil(minsz / 2)
        mide = np.floor((minsz - 1) / 2) - 1

        resizeddft[0:mids[0], 0:mids[1], :, :] = scaling * inputdft[0:mids[0], 0:mids[1], :, :]
        resizeddft[0:mids[0], (-1 - mide[1])::, :, :] = scaling * inputdft[0:mids[0], (-1 - mide[1])::, :, :]
        resizeddft[(-1 - mide[0])::, 0:mids[1], :, :] = scaling * inputdft[(-1 - mide[0])::, 0:mids[1], :, :]
        resizeddft[(-1 - mide[0])::, (-1 - mide[1])::, :, :] = scaling * \
                                                               inputdft[(-1 - mide[0])::, (-1 - mide[1])::, :, :]
    else:
        resizeddft = inputdft

    return resizeddft