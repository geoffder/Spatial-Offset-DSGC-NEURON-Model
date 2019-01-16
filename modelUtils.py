from neuron import h
import numpy as np
from scipy.signal import find_peaks


def frange(start, stop, step):
    '''iterator for floats'''
    i = start
    while i < stop:
        # makes this work as an iterator, returns value then continues
        # loop for next call of function
        yield i
        i += step


def findOrigin(dends):
    '''find the centre point of the cell/arbour'''
    leftX = 1000
    rightX = -1000
    topY = -1000
    botY = 1000
    for i in range(len(dends)):
        dends[i].push()

        # mid point
        pts = int(h.n3d())  # number of 3d points of section
        if(pts % 2):  # odd number of points
            xLoc = h.x3d((pts-1)/2)
            yLoc = h.y3d((pts-1)/2)
        else:
            xLoc = (h.x3d(pts/2)+h.x3d((pts/2)-1))/2.0
            yLoc = (h.y3d(pts/2)+h.y3d((pts/2)-1))/2.0
        if (xLoc < leftX):
            leftX = xLoc
        if (xLoc > rightX):
            rightX = xLoc
        if (yLoc < botY):
            botY = yLoc
        if (yLoc > topY):
            topY = yLoc

        # terminal point
        xLoc = h.x3d(pts-1)
        yLoc = h.y3d(pts-1)
        if (xLoc < leftX):
            leftX = xLoc
        if (xLoc > rightX):
            rightX = xLoc
        if (yLoc < botY):
            botY = yLoc
        if (yLoc > topY):
            topY = yLoc

        h.pop_section()
    # print  'leftX: '+str(leftX)+ ', rightX: '+str(rightX)
    # print  'topY: '+str(topY)+ ', botY: '+str(botY)
    return (leftX+(rightX-leftX)/2, botY+(topY-botY)/2)


def rotate(origin, X, Y, angle):
    """
    Rotate a points (X[i],Y[i]) counterclockwise an angle around an origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    X, Y = np.array(X), np.array(Y)
    rotX = ox + np.cos(angle) * (X - ox) - np.sin(angle) * (Y - oy)
    rotY = oy + np.sin(angle) * (X - ox) + np.cos(angle) * (Y - oy)
    return rotX, rotY


def findSpikes(Vm, thresh=20):
    '''use scipy.signal.find_peaks to get spike count and times'''
    spikes, _ = find_peaks(Vm, height=thresh)  # returns indices
    count = spikes.size
    times = spikes * h.dt

    return count, times
