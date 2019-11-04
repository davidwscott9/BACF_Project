def computePascalScore(bb1, bb2):
    from computeArea import computeArea
    from computeIntersectionArea import computeIntersectionArea
    # compute the Pascal score of the bb1, bb2 (intersection/union)
    intersectionArea = computeIntersectionArea(bb1,bb2)
    pascalScore = intersectionArea/(computeArea(bb1)+computeArea(bb2)-intersectionArea)

    return pascalScore