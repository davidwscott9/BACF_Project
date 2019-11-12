def fhog_dlib(I, binSize, nOrients, clip, crop):
    import dlib

    H = dlib.fhog_object_detector.run()

    return H