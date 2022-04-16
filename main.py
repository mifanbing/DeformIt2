from Data.DataLoader import DataLoader
from Data.Util import Util
import numpy as np
import cv2
import math

inputImageName = "lbj.png"
inWidth = 500
inHeight = 500

dataLoader = DataLoader(inWidth, inHeight, inputImageName)
inputContourPoints = dataLoader.getContourPoints()
posePoints = dataLoader.posePoints
poseLines = dataLoader.poseLines

util = Util(inWidth, inHeight)
inputContourPointsRefine = []
for i in range(0, len(inputContourPoints) - 1):
  for point in util.getInterpolatePoints(inputContourPoints[i], inputContourPoints[i+1]):
    inputContourPointsRefine.append(point)
    
workImage = np.zeros((inHeight, inWidth, 3), dtype = np.uint8)

bodyContour = util.getBodyContour([poseLines[3], poseLines[5], poseLines[8], poseLines[11]], inputContourPointsRefine)
util.drawContour(bodyContour, workImage, dataLoader.inputImageResize)
# for point in bodyContour:
#     ww, hh = point
#     workImage[hh, ww] = (0, 0, 255)
      

rotatedLeftArmContour = util.rotateContour(poseLines[3], inputContourPointsRefine, 0, math.pi / 1.5, workImage, dataLoader.inputImageResize)
# for point in rotatedLeftArmContour:
#     ww, hh = point
#     workImage[hh, ww] = (0, 255, 255)
    
rotatedRightArmContour = util.rotateContour(poseLines[5], inputContourPointsRefine, math.pi / 4, -math.pi/2.5, workImage, dataLoader.inputImageResize)
# for point in rotatedRightArmContour:
#     ww, hh = point
#     workImage[hh, ww] = (0, 0, 255)
    
rotatedLeftLegContour = util.rotateContour(poseLines[8], inputContourPointsRefine, math.pi / 6, -math.pi / 6, workImage, dataLoader.inputImageResize)
# for point in rotatedLeftLegContour:
#     ww, hh = point
#     workImage[hh, ww] = (0, 0, 255)
    
rotatedRightLegContour = util.rotateContour(poseLines[11], inputContourPointsRefine, -math.pi / 6, -math.pi / 6, workImage, dataLoader.inputImageResize)
# for point in rotatedRightLegContour:
#     ww, hh = point
#     workImage[hh, ww] = (0, 0, 255)    

cv2.imshow('', workImage)
cv2.waitKey(0)





