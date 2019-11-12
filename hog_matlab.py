def hog_matlab(k):
    import numpy as np
    import scipy.io as sio

    if k == 0:
        hog_image = sio.loadmat('hog_image.mat')
        hog_image = np.array(hog_image['hog_image'])

    if k == 1:
        hog_image = sio.loadmat('hog_image1.mat')
        hog_image = np.array(hog_image['hog_image1'])

    if k == 2:
        hog_image = sio.loadmat('hog_image2.mat')
        hog_image = np.array(hog_image['hog_image2'])
    if k == 3:
        hog_image = sio.loadmat('hog_image3.mat')
        hog_image = np.array(hog_image['hog_image3'])

    if k == 4:
        hog_image = sio.loadmat('hog_image4.mat')
        hog_image = np.array(hog_image['hog_image4'])

    if k == 5:
        hog_image = sio.loadmat('hog_image5.mat')
        hog_image = np.array(hog_image['hog_image5'])

    return hog_image
