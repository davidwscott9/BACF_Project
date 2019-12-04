def computeIntersectionArea(bb1, bb2):
    """
    Computes the intersecting area of two bounding boxes
    :param bb1: a list of the form [xmin, ymin, xmax, ymax]
    :param bb2: a list of the form [xmin, ymin, xmax, ymax]
    :return: the area of intersection
    """
    from computeArea import computeArea

    xmin = max(bb1[0], bb2[0])
    xmax = min(bb1[2], bb2[2])
    ymin = max(bb1[1], bb2[1])
    ymax = min(bb1[3], bb2[3])
    areaIntersection = computeArea([xmin, ymin, xmax, ymax])

    return areaIntersection
