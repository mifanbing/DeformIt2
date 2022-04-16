import cv2
import math

class DataLoader:
    canny_low = 30
    canny_high = 200
    
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13]]
    pointsLBJNormal = [
      (0.510,	0.164),  # 0: head
      (0.512,	0.311),  # 1: chest
      (0.419,	0.264),  # 2: left shoulder
      (0.356,	0.242), # 3: left elbow
      (0.245,	0.12), # 4: left hand
      (0.60,	0.271), # 5: right shoulder
      (0.644,	0.254), # 6: right elbow
      (0.749,	0.14), # 7: right hand
      (0.442,	0.63), # 8: left waist
      (0.436,	0.721), # 9: left knee
      (0.415,	0.933), # 10: left foot
      (0.520,	0.63), # 11: right waist
      (0.525,	0.724), # 12: right knee
      (0.507,	0.933)] # 13: right foot
    
    def __init__(self, inWidth, inHeight, imageName):  
        self.inWidth = inWidth
        self.inHeight = inHeight
        self.imageName = imageName
        self.posePoints = []
        for pair in self.pointsLBJNormal:
          wNormal, hNormal = pair
          ww = math.floor(wNormal * self.inWidth)
          hh = math.floor(hNormal * inHeight)
          self.posePoints.append((ww, hh))
        self.poseLines = []
        for pair in self.POSE_PAIRS:
          indexStart = pair[0]
          indexEnd = pair[1]
        
          self.poseLines.append((self.posePoints[indexStart], self.posePoints[indexEnd]))
        inputImage = cv2.imread(self.imageName)
        self.inputImageResize = cv2.resize(inputImage, (self.inWidth, self.inHeight))
        
        
    def getContourPoints(self):
      
      # Convert image to grayscale        
      image_gray = cv2.cvtColor(self.inputImageResize, cv2.COLOR_BGR2GRAY)
      # Apply Canny Edge Dection
      edges = cv2.Canny(image_gray, self.canny_low, self.canny_high)
      edges = cv2.dilate(edges, None)
      edges = cv2.erode(edges, None)
      # get the contours and their areas
      _, inputContours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
      inputMaxContour = inputContours[0]
      for contour in inputContours:
        if cv2.contourArea(contour) > cv2.contourArea(inputMaxContour):
          inputMaxContour = contour
    
      inputContourPoints = inputMaxContour[:, 0, :]
      
      #mutate so that index 0 is head
      inputContourPointsMutate = []
      for point in inputContourPoints[1550:]:
        inputContourPointsMutate.append(point)
            
      for point in inputContourPoints[:1550]:
        inputContourPointsMutate.append(point)
    
      return inputContourPointsMutate
                           