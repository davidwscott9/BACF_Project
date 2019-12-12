def fhog(I, binSize, nOrients, clip, crop):

    from gradientMag import gradientMag
    from gradientHist import gradientHist
    if binSize is None:
        binSize = 8
    if nOrients is None:
        nOrients = 9
    if clip is None:
        clip = 0.2
    if crop is None:
        crop = 0

    softBin = -1
    useHog = 2
    b = binSize

    ## NEED TO IMPORT THESE NEXT FUNCTIONS FROM ANOTHER FILE ##
    [M, O] = gradientMag(I, 0, 0, 0, 1)
    H = gradientHist(M, O, binSize, nOrients, softBin, useHog, clip)

    if crop > 0: # if running into problems, this may be a source. Not sure what MATLAB function is doing here
        # this should output a 1x2 array of two values to see if even or odd.
        e = I.shape%b < b / 2 # need to see how this works with np arrays (or whatever OpenCV uses for images)
        H= H[2:-1-e[0], 2:-1-e[1],:]

    return H
