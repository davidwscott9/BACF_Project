def fhog_python(I, binSize, nOrients, clip, crop):

    import cv2
    hog = cv2.HOGDescriptor()
    H = hog.compute(I)

    return H
