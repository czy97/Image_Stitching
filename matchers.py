import cv2
import numpy as np


class matchers:
	def __init__(self,detector = 'SURF',descriptor = 'BRIEF'):
		if(detector=='SURF'):
			self.detector = cv2.xfeatures2d.SURF_create()
		elif(detector=='SIFT'):
			self.detector = cv2.xfeatures2d.SIFT_create()
		elif(detector=='ORB'):
			self.detector = cv2.ORB_create()

		if(descriptor == 'BRIEF'):
			self.descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes = 32,use_orientation = True)
		else:
			self.descriptor = None

		self.binary = False
		if(descriptor == 'BRIEF' or descriptor == 'ORB'):
			self.binary = True

		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm=0, trees=5)
		search_params = dict(checks=50)
		if(not self.binary):
			self.flann = cv2.FlannBasedMatcher(index_params, search_params)
		else:
			self.flann = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


	def match(self, i1, i2, direction=None):
		imageSet1 = self.getFeatures(i1)
		imageSet2 = self.getFeatures(i2)

		self.kp1 = imageSet1['kp']
		self.kp2 = imageSet2['kp']
		self.img1 = i1
		self.img2 = i2
		self.goodMatches = []

		if(not self.binary):
			matches = self.flann.knnMatch(imageSet2['des'],imageSet1['des'],k=2)
			good = []
			for i, (m, n) in enumerate(matches):
				if m.distance < 0.7 * n.distance:
					good.append((m.trainIdx, m.queryIdx))
					self.goodMatches.append(m)
		else:
			matches = self.flann.match(imageSet2['des'],imageSet1['des'])
			matches = sorted(matches, key=lambda x: x.distance)
			good = []
			for i, m in enumerate(matches):
				good.append((m.trainIdx, m.queryIdx))
			self.goodMatches = matches

		if len(good) > 4:
			pointsCurrent = imageSet2['kp']
			pointsPrevious = imageSet1['kp']

			matchedPointsCurrent = np.float32(
				[pointsCurrent[i].pt for (__, i) in good]
			)
			matchedPointsPrev = np.float32(
				[pointsPrevious[i].pt for (i, __) in good]
				)

			H, s = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 4)
			return H
		return None

	def getFeatures(self, im):
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		kp = self.detector.detect(gray)

		if(self.descriptor == None):
			kp, des = self.detector.compute(gray, kp)
		else:
			kp, des = self.descriptor.compute(gray, kp)



		return {'kp':kp, 'des':des}
	def drawMatch(self,filepath = 'match.jpg'):
		img3 = None
		img3 = cv2.drawMatches(self.img2, self.kp2, self.img1, self.kp1, self.goodMatches[:20], img3,flags=2,)
		cv2.imwrite(filepath, img3)
		# cv2.imwrite(STORE_MATCH, img3)