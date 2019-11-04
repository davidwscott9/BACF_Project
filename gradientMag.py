def gradientMag(I, channel, normRad, normConst, full):
    if I is None or len(I) == 0:
        M = []
        O = M
    if channel is None or len(channel) == 0:
        channel = 0
    if normRad is None or len(normRad) == 0:
        normRad = 0
    if normConst is None or len(normConst) == 0:
        normConst = 0.005
    if full is None or len(full) == 0:
        full = 0

    if I is not None:
        M = gradientMex('gradientMag', I, channel, full)
    else:
        [M, O] = gradientMex('gradientMag', I, channel, full)

    if normRad == 0:
        return
    else:
        S = convTri(M, normRad)
        gradientMex('gradientMagNorm', M, S, normConst)  # operates on M - WHAT DOES THIS DO???
        return M, O