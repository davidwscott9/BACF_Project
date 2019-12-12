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

    if k == 6:
        hog_image = sio.loadmat('hog_image6.mat')
        hog_image = np.array(hog_image['hog_image6'])

    if k == 7:
        hog_image = sio.loadmat('hog_image7.mat')
        hog_image = np.array(hog_image['hog_image7'])

    if k == 8:
        hog_image = sio.loadmat('hog_image8.mat')
        hog_image = np.array(hog_image['hog_image8'])
    if k == 9:
        hog_image = sio.loadmat('hog_image9.mat')
        hog_image = np.array(hog_image['hog_image9'])

    if k == 10:
        hog_image = sio.loadmat('hog_image10.mat')
        hog_image = np.array(hog_image['hog_image10'])

    if k == 11:
        hog_image = sio.loadmat('hog_image11.mat')
        hog_image = np.array(hog_image['hog_image11'])

    if k == 12:
        hog_image = sio.loadmat('hog_image12.mat')
        hog_image = np.array(hog_image['hog_image12'])
    return hog_image
