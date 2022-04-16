import numpy as np
import math
import cv2
from numpy.linalg import inv

class Util:
    def __init__(self, inWidth, inHeight):
        self.inWidth = inWidth
        self.inHeight = inHeight
        self.cutContours = [[]] * 8
        
    def findStartAndEnd(self, contourPoints, targetPoint, startPoint, endPoint):
      w, h = targetPoint
      wStart, hStart = startPoint
      wEnd, hEnd = endPoint
    
      left = -1
      right = self.inWidth
      startIndex = -1
      endIndex = -1
    
      index90Degree = []
    
      for i in range(len(contourPoints)):
        point = contourPoints[i]
        wPoint, hPoint = point
    
        length1 = math.sqrt((hEnd - hStart) ** 2 + (wEnd - wStart) ** 2)
        length2 = math.sqrt((hPoint - h) ** 2 + (wPoint - w) ** 2)
        if length2 == 0:
          continue
        dotProduct = (hEnd - hStart) * (hPoint - h) + (wEnd - wStart) * (wPoint - w)
        angle = abs(np.arccos(dotProduct / length1 / length2))
        if abs(angle - math.pi / 2) < math.pi / 20:
          if startIndex == -1:
            startIndex = i
          else:
            endIndex = i
        else:
          if startIndex != -1 and endIndex != -1:
            index90Degree.append((startIndex, endIndex))
            startIndex = -1
            endIndex = -1
  
      # the 90 degree cut for left leg can include the right leg
      index90DegreeRefine = []
      left = -1
      right = self.inWidth
      index90DegreeLeft = -1, -1
      index90DegreeRight = -1, -1
      for pair in index90Degree:
        start2, end2 = pair
        ww, hh = contourPoints[start2]
        if ww < w and ww > left:
          left = contourPoints[start2][0]
          index90DegreeLeft = pair
        if ww > w and ww < right:
          right = contourPoints[start2][0]
          index90DegreeRight = pair
    
      index90DegreeRefine = [index90DegreeLeft, index90DegreeRight] if index90DegreeLeft[0] < index90DegreeRight[0] else [index90DegreeRight, index90DegreeLeft]
      return index90DegreeRefine
    
    def getInterpolatePoints(self, pointStart, pointEnd):
      wStart, hStart = pointStart
      wEnd, hEnd = pointEnd
    
      points = []
    
      if abs(wStart - wEnd) > abs(hStart - hEnd):
        step =  1 if wStart < wEnd else -1
    
        for w in range(wStart, wEnd, step):
          k = (w - wStart) / (wEnd - wStart)
          h = round(hStart + k * (hEnd - hStart))
          points.append((w, h))
      else:
        step =  1 if hStart < hEnd else -1
    
        for h in range(hStart, hEnd, step):
          k = (h - hStart) / (hEnd - hStart)
          w = round(wStart + k * (wEnd - wStart))
          points.append((w, h))
    
      return points

    def getPartContour(self, poseLine, contourPoints, index):
        pointPartStart, pointPartEnd = poseLine
        
        startPointRange = self.findStartAndEnd(contourPoints, pointPartStart, pointPartStart, pointPartEnd)
        startPointStartIndex = startPointRange[0][0]
        startPointEndIndex = startPointRange[1][0]
        startPointCutContour = self.getInterpolatePoints(contourPoints[startPointStartIndex], contourPoints[startPointEndIndex])
        
        endPointRange = self.findStartAndEnd(contourPoints, pointPartEnd, pointPartStart, pointPartEnd)
        endPointStartIndex = endPointRange[0][0]
        endPointEndIndex = endPointRange[1][0]
        endPointCutContour = self.getInterpolatePoints(contourPoints[endPointStartIndex], contourPoints[endPointEndIndex])
        
        self.cutContours[index] = startPointCutContour
        self.cutContours[index + 1] = endPointCutContour
        
        partUpperContour = []
        partUpperContour.extend(startPointCutContour)
        
        # print("startPoint startIndex: %d endIndex: %d" % (startPointStartIndex, startPointEndIndex))
        # print("endPoint startIndex: %d endIndex: %d" % (endPointStartIndex, endPointEndIndex))
        
        partUpperContour.extend(contourPoints[endPointEndIndex:startPointEndIndex])
        partUpperContour.extend(endPointCutContour)
        partUpperContour.extend(contourPoints[startPointStartIndex:endPointStartIndex])
        
        partLowerContour = []
        partLowerContour.extend(endPointCutContour)
        partLowerContour.extend(contourPoints[endPointStartIndex:endPointEndIndex])
    
        return partUpperContour, partLowerContour
    
    def getBodyContour(self, poseLines, contourPoints):
        def inInterval(target, start, end, length):
          if end - start < length / 2:
            return target > start and target < end
          else:
            return target > end or target < start

        intervals = []
        cutContours = []
        for poseLine in poseLines:
            pointPartStart, pointPartEnd = poseLine
            rangeStartAndEnd = self.findStartAndEnd(contourPoints, pointPartStart, pointPartStart, pointPartEnd)
            intervals.append((rangeStartAndEnd[0][0], rangeStartAndEnd[1][0]))
            cutContours.append(self.getInterpolatePoints(contourPoints[rangeStartAndEnd[0][0]], contourPoints[rangeStartAndEnd[1][0]]))
        
        
        trimmedBodyContour = []
        hasAddedParts = [False, False, False, False]
        for i in range(len(contourPoints)):
            isBody = True
            for j in range(len(hasAddedParts)):
                if inInterval(i, intervals[j][0], intervals[j][1], len(contourPoints)):
                    isBody = False
                    if not hasAddedParts[j]:
                        hasAddedParts[j] = True
                        trimmedBodyContour.extend(cutContours[j])
            if isBody:
                trimmedBodyContour.append(contourPoints[i])
                
        return trimmedBodyContour

    def rotateContour(self, poseLine, contourPoints, angleUpper, angleLower, workImage, inputImage):
        pointPartStart, pointPartEnd = poseLine
        
        startPointRange = self.findStartAndEnd(contourPoints, pointPartStart, pointPartStart, pointPartEnd)
        startPointStartIndex = startPointRange[0][0]
        startPointEndIndex = startPointRange[1][0]
        startPointCutContour = self.getInterpolatePoints(contourPoints[startPointStartIndex], contourPoints[startPointEndIndex])
        
        endPointRange = self.findStartAndEnd(contourPoints, pointPartEnd, pointPartStart, pointPartEnd)
        endPointStartIndex = endPointRange[0][0]
        endPointEndIndex = endPointRange[1][0]
        endPointCutContour = self.getInterpolatePoints(contourPoints[endPointStartIndex], contourPoints[endPointEndIndex])
        
        w0, h0 = contourPoints[startPointStartIndex]
        wEndPointStart, hEndPointStart = contourPoints[endPointStartIndex]
        wEndPointEnd, hEndPointEnd = contourPoints[endPointEndIndex]
        
        w2 = int((wEndPointEnd - w0) * math.cos(angleUpper) - (hEndPointEnd - h0) * math.sin(angleUpper) + w0)
        h2 = int((wEndPointEnd - w0) * math.sin(angleUpper) + (hEndPointEnd - h0) * math.cos(angleUpper) + h0)
        w3 = int((wEndPointStart - w0) * math.cos(angleUpper) - (hEndPointStart - h0) * math.sin(angleUpper) + w0)
        h3 = int((wEndPointStart - w0) * math.sin(angleUpper) + (hEndPointStart - h0) * math.cos(angleUpper) + h0)
        
        rotatedContour = []
        lowerContour = []
        
        for point in startPointCutContour:
            rotatedContour.append(point)
            
        # upper part
        for point in contourPoints[startPointStartIndex:endPointStartIndex]:
            w, h = point
            wRotate = int((w - w0) * math.cos(angleUpper) - (h - h0) * math.sin(angleUpper) + w0)
            hRotate = int((w - w0) * math.sin(angleUpper) + (h - h0) * math.cos(angleUpper) + h0)
            rotatedContour.append((wRotate, hRotate))
            
        wRotateLower, hRotateLower = rotatedContour[-1]
        for point in endPointCutContour:
            w, h = point
            wRotate = int((w - w0) * math.cos(angleUpper) - (h - h0) * math.sin(angleUpper) + w0)
            hRotate = int((w - w0) * math.sin(angleUpper) + (h - h0) * math.cos(angleUpper) + h0)
            wRotate2 = int((wRotate - wRotateLower) * math.cos(angleLower) - (hRotate - hRotateLower) * math.sin(angleLower) + wRotateLower)
            hRotate2 = int((wRotate - wRotateLower) * math.sin(angleLower) + (hRotate - hRotateLower) * math.cos(angleLower) + hRotateLower)
            rotatedContour.append((wRotate2, hRotate2))
            lowerContour.append((wRotate2, hRotate2))
        
        controlPointsInput = [contourPoints[startPointStartIndex], contourPoints[startPointEndIndex], contourPoints[endPointEndIndex]]
        controlPointsOutput = [contourPoints[startPointStartIndex], contourPoints[startPointEndIndex], (w2, h2)]
  
        for point in contourPoints[endPointEndIndex:startPointEndIndex]:
            wRotate, hRotate = self.mapPoint(point, controlPointsInput, controlPointsOutput)
            rotatedContour.append((wRotate, hRotate)) 
        
        upperContourInput = [contourPoints[startPointStartIndex], contourPoints[startPointEndIndex], contourPoints[endPointStartIndex], contourPoints[endPointEndIndex]]
        upperContourOutput = [contourPoints[startPointStartIndex], contourPoints[startPointEndIndex], (w3, h3), (w2, h2)]
        upperContourRefine = []
        rotatedContour.append(rotatedContour[0])
        for i in range(0, len(rotatedContour) - 1):
          for point in self.getInterpolatePoints(rotatedContour[i], rotatedContour[i+1]):
            upperContourRefine.append(point) 
        self.drawMappedContour(upperContourRefine, workImage, inputImage, upperContourOutput, upperContourInput)
        
        
        #lower part
        for point in contourPoints[endPointStartIndex:endPointEndIndex]:
            w, h = point
            wRotate = int((w - w0) * math.cos(angleUpper) - (h - h0) * math.sin(angleUpper) + w0)
            hRotate = int((w - w0) * math.sin(angleUpper) + (h - h0) * math.cos(angleUpper) + h0)
            wRotate2 = int((wRotate - wRotateLower) * math.cos(angleLower) - (hRotate - hRotateLower) * math.sin(angleLower) + wRotateLower)
            hRotate2 = int((wRotate - wRotateLower) * math.sin(angleLower) + (hRotate - hRotateLower) * math.cos(angleLower) + hRotateLower)
            rotatedContour.append((wRotate2, hRotate2))
            lowerContour.append((wRotate2, hRotate2))
        lowerContour.append(lowerContour[0])
        
        lowerContourRefine = []
        for i in range(0, len(lowerContour) - 1):
          for point in self.getInterpolatePoints(lowerContour[i], lowerContour[i+1]):
            lowerContourRefine.append(point)    
        self.drawRotatedContour(lowerContourRefine, workImage, inputImage, -angleUpper, w0, h0, -angleLower, wRotateLower, hRotateLower)
        
        return upperContourRefine

    def mapPoint(self, point, controlPointsInput, controlPointsOutput):
        w, h = point
        w1Input, h1Input = controlPointsInput[0]
        w2Input, h2Input = controlPointsInput[1]
        w3Input, h3Input = controlPointsInput[2]
        w1Output, h1Output = controlPointsOutput[0]
        w2Output, h2Output = controlPointsOutput[1]
        w3Output, h3Output = controlPointsOutput[2]
        
        M = [[w1Input, w2Input, w3Input],
             [h1Input, h2Input, h3Input],
             [1, 1, 1]]
        MInv = inv(M)
        weights = np.matmul(MInv, [[w], [h], [1]])
        
        wMap = int(w1Output * weights[0][0] + w2Output * weights[1][0] + w3Output * weights[2][0])
        hMap = int(h1Output * weights[0][0] + h2Output * weights[1][0] + h3Output * weights[2][0])
        
        return wMap, hMap
    
    def findMapPoint(self, point, upperContourInput, upperContourOutput):
        #triangle1: startPointStart startPointEnd endPointStart
        #triangle2: endPointStart, endPointEnd startPointEnd
        #line: endPointStart startPointEnd
        startPointStart = upperContourInput[0]
        startPointEnd = upperContourInput[1]
        endPointStart = upperContourInput[2]
        #endPointEnd = upperContourInput[3]
        
        #line: ax + y + c = 0
        x1, y1 = startPointEnd
        x2, y2 = endPointStart
        a = (y2 - y1) / (x1 - x2)
        c = (y2 * x1 - y1 * x2) / (x2 - x1)
        x, y = point
        xStart, yStart = startPointStart
        startPointStartIsAbove = a * xStart + yStart + c 
        pointIsAbove = a * x + y + c 
        
        if startPointStartIsAbove * pointIsAbove > 0:
            controlPointsInput = [upperContourInput[0], upperContourInput[1], upperContourInput[2]]
            controlPointsOutput = [upperContourOutput[0], upperContourOutput[1], upperContourOutput[2]]
            return self.mapPoint(point, controlPointsInput, controlPointsOutput)
        else:
            controlPointsInput = [upperContourInput[1], upperContourInput[2], upperContourInput[3]]
            controlPointsOutput = [upperContourOutput[1], upperContourOutput[2], upperContourOutput[3]]
            return self.mapPoint(point, controlPointsInput, controlPointsOutput)

    def drawRotatedContour(self, contour, workImage, inputImage, angleUpper, w0, h0, angleLower, wLower, hLower):
      hMin = self.inHeight
      hMax = -1
      for point in contour:
        w, h = point
        if h < hMin:
          hMin = h
        if h > hMax:
          hMax = h
    
      for h in range(hMin, hMax):
        wMin = self.inWidth
        wMax = -1
        for point in contour:
          w, h2 = point
          if h2 == h:
            if w < wMin:
              wMin = w
            if w > wMax:
              wMax = w
        for w in range(wMin, wMax):
          
          wRotate2 = int((w - wLower) * math.cos(angleLower) - (h - hLower) * math.sin(angleLower) + wLower)
          hRotate2 = int((w - wLower) * math.sin(angleLower) + (h - hLower) * math.cos(angleLower) + hLower)
          wRotate = int((wRotate2 - w0) * math.cos(angleUpper) - (hRotate2 - h0) * math.sin(angleUpper) + w0)
          hRotate = int((wRotate2 - w0) * math.sin(angleUpper) + (hRotate2 - h0) * math.cos(angleUpper) + h0)
            
          workImage[h, w] = inputImage[hRotate, wRotate]   
    
    def drawMappedContour(self, contour, workImage, inputImage, upperContourInput, upperContourOutput):
      hMin = self.inHeight
      hMax = -1
      for point in contour:
        w, h = point
        if h < hMin:
          hMin = h
        if h > hMax:
          hMax = h
    
      for h in range(hMin, hMax):
        wMin = self.inWidth
        wMax = -1
        for point in contour:
          w, h2 = point
          if h2 == h:
            if w < wMin:
              wMin = w
            if w > wMax:
              wMax = w
        for w in range(wMin, wMax):
          wInput, hInput = self.findMapPoint((w, h), upperContourInput, upperContourOutput)
          workImage[h, w] = inputImage[hInput, wInput]            
        
        
    def drawContour(self, contour, workImage, inputImage):
      hMin = self.inHeight
      hMax = -1
      for point in contour:
        w, h = point
        if h < hMin:
          hMin = h
        if h > hMax:
          hMax = h
    
      for h in range(hMin, hMax):
        wMin = self.inWidth
        wMax = -1
        for point in contour:
          w, h2 = point
          if h2 == h:
            if w < wMin:
              wMin = w
            if w > wMax:
              wMax = w
        for w in range(wMin, wMax):
          workImage[h, w] = inputImage[h, w]         

        
        
        
        
    
  
    
  
    
  
    
  