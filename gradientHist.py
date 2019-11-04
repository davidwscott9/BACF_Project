def gradientHist(M, O, binSize, nOrients, softBin, useHog, clip):
    import cv2
    hog = cv2.HOGDescriptor()
    H = hog.compute(M)

    return H
