def computeArea(bb):
    """
    This function computes the area of the bounding box, bb.
    :param bb: the bounding box in the format bb = [xmin ymin xmax ymax]
    :return: the area of the bounding box
    """

    if bb[0] > bb[2] or bb[1] > bb[3]:
        areaBB = 0
    else:
        areaBB = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1)

    return areaBB
