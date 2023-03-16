import numpy as np
import random
import math

width = 1
height = 1
midPoint = [1/2, 1/2]
pointA_Left = [0.05, 0.05] 
pointB_Right = [0.95, 0.05]
pointC_Leg = [0.5, 0.95]
miceHead = []
miceBody = miceHead
miceTail = miceHead
pack =[midPoint, pointA_Left, pointB_Right, pointC_Leg]

#initialize 5 frames 
frames = np.zeros((20,7,2))

for frame in frames:
    for i in range(0, 4):
        frame[i] = pack[i]

#populate micepoints
for i in range(0, frames.shape[0]):
    pt = [round((random.uniform(0, 1)), 2), round((random.uniform(0, 1)), 2)]
    miceHead.append(pt)

#add micepoints to frame
x=0
for frame in frames:
    frame[4] = miceHead[x]
    x += 1

def lerp(p1, p2, f):
    reciprocal = 1 - f
    startPoint = p1 * reciprocal
    EndPoint = p2 * f
    ScaledBy_f = (startPoint + EndPoint)
    return ScaledBy_f

#Define mid entry for the 3 points(points <= 30%)
def mid_point_entry(percentage):
    value = []
    Ax = lerp(pointA_Left[0], midPoint[0], percentage)
    Ay = lerp(pointA_Left[1], midPoint[1], percentage)
    Bx = lerp(pointB_Right[0], midPoint[0], percentage)
    By = lerp(pointB_Right[1], midPoint[1], percentage)
    Cx = lerp(pointC_Leg[0], midPoint[0], percentage)
    Cy = lerp(pointC_Leg[1], midPoint[1], percentage)
    value =[[Ax, Ay], [Bx, By], [Cx, Cy]]
    return (value)
midA,midB,midC = mid_point_entry(0.70)

#output mouse locatio
armVisit = []
miceInMid = []
for frame in frames:
    miceX = frame[4][0]
    miceY = frame[4][1]

    #test A
    if (miceX <= midPoint[0] and miceY <= midPoint[1]):
        if miceX  >= midA[0] and miceY >= midA[1]:
            miceInMid.append('M')
        else:
            armVisit.append('A')

    #test B
    if (miceX >= midPoint[0] and miceY <= midPoint[1]):
        if miceX  >= midB[0] and miceY >= midB[1]:
            miceInMid.append('M')
        else:
            armVisit.append('B')

    #test C
    if (miceY >= midPoint[1]):
        if (miceY <= midC[1]):
            miceInMid.append('M')
        else:
            armVisit.append('C')
    
print(miceInMid)

#determine SAP...
SAR = []
AAR = []
SAP = []
x=armVisit

for i in range(0,len(x)-1):
    if x[i] == x[i+1]:
       SAR.append([x[i],x[i+1]])
        
for i in range(0,len(x)-2):
    if (
        x[i] == x[i+1] and x[i] != x[i+2] or
        x[i] == x[i+2] and x[i] != x[i+1] or
        x[i+1] == x[i+2] and x[i+1] != x[i]
        ):
        AAR.append([x[i],x[i+1],x[i+2]])
    elif (
        x[i] != x[i+1] and x[i+1] != [x[i+2]] and x[i+1] != [x[i]]
        ):
        SAP.append([x[i],x[i+1],x[i+2]])

print(armVisit)
print("SAR:{}".format(SAR))
print("AAR{}".format(AAR))
print("SAP{}".format(SAP))
