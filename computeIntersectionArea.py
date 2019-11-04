def computeIntersectionArea(bb1, bb2):
    from computeArea import computeArea
    # compute intersection anrea of bb1 and bb2
    # bb1 and bb2 - bounding boxes
    # bbi = [xmin ymin xmax ymax] for i=1,2

    xmin = max(bb1[0], bb2[0])
    xmax = min(bb1[2], bb2[2])
    ymin = max(bb1[1], bb2[1])
    ymax = min(bb1[3], bb2[3])
    areaIntersection = computeArea([xmin, ymin, xmax, ymax])

    return areaIntersection
