def computeArea(bb):
    # computes area of the bb = [xmin ymin xmax ymax]

    if bb[0] > bb[2] or bb[1] > bb[3]:
        areaBB = 0
    else:
        areaBB = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1)

    return areaBB