def computePascalScore(bb1, bb2):
    """
    Computes the Intersection over Union (IoU) also known as the Pascal Score of two bounding boxes
    :param bb1: a list of the form [xmin ymin xmax ymax]
    :param bb2: a list of the form [xmin ymin xmax ymax]
    :return: the score of the bounding boxes
    """
    from computeArea import computeArea
    from computeIntersectionArea import computeIntersectionArea
    # compute the Pascal score of the bb1, bb2 (intersection/union)
    intersectionArea = computeIntersectionArea(bb1,bb2)

    if (computeArea(bb1)+computeArea(bb2)-intersectionArea) == 0:
        pascalScore = 0
    else:
        pascalScore = intersectionArea/(computeArea(bb1)+computeArea(bb2)-intersectionArea)

    return pascalScore