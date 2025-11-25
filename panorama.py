import numpy as np
import imutils
import cv2
class Stitcher:
	def __init__(self):
		# Xác định xem ta có đang sử dụng OpenCV v3.X hay không
		self.isv3 = imutils.is_cv3(or_better=True)

	def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
		# Giải nén hình ảnh, sau đó phát hiện các điểm chính (keypoint) và trích xuất các mô tả bất biến cục bộ
		(imageB, imageA) = images
		(kpsA, featuresA) = self.detectAndDescribe(imageA)
		(kpsB, featuresB) = self.detectAndDescribe(imageB)
		imageA_key = cv2.drawKeypoints(imageA, [cv2.KeyPoint(x=float(x), y=float(y), size=3) for (x, y) in kpsA], None, (0,255,0))
		imageB_key = cv2.drawKeypoints(imageB, [cv2.KeyPoint(x=float(x), y=float(y), size=3) for (x, y) in kpsB], None, (0,255,0))
		# Khớp các tính năng (feature) giữa hai hình ảnh
		M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
		# Nếu kết quả khớp là None thì không có đủ kết quả khớp keypoint để tạo ra một bức tranh toàn cảnh
		if M is None:
			return None
		# Nếu không, hãy áp dụng hiệu ứng cong vênh để ghép các hình ảnh cùng nhau
		(matches, H, status) = M
		result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
		result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
		# Kiểm tra xem các keypoint có khớp nhau hay không để trực quan hóa
		if showMatches:
			vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
			# Trả về một bộ hình ảnh được khâu và hình dung
			return (result, vis, imageA_key, imageB_key)
		# Trả về một bộ hình ảnh đã khâu
		return result

	def detectAndDescribe(self, image):
		# Chuyển đổi hình ảnh sang thang độ xám
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# Kiểm tra xem ta có đang sử dụng OpenCV v3.X hay không
		if self.isv3:
			# Phát hiện và trích xuất các đặc điểm từ hình ảnh
			descriptor = cv2.xfeatures2d.SIFT_create()
			(kps, features) = descriptor.detectAndCompute(image, None)
		# Nếu không, ta đang sử dụng OpenCV 2.4.X
		else:
			# Phát hiện các keypoint trong hình ảnh
			detector = cv2.FeatureDetector_create("SIFT")
			kps = detector.detect(gray)
			# Trích xuất các feature từ hình ảnh
			extractor = cv2.DescriptorExtractor_create("SIFT")
			(kps, features) = extractor.compute(gray, kps)
		# Chuyển đổi các keypoint từ đối tượng KeyPoint sang mảng NumPy
		kps = np.float32([kp.pt for kp in kps])
		# Trả về một bộ các keypoint và các feature
		return (kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
		# Tính toán các kết quả khớp thô (raw matches) và khởi tạo danh sách các kết quả khớp thực tế
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []
		# Lặp qua các kết quả khớp thô
		for m in rawMatches:
			# Đảm bảo khoảng cách nằm trong một tỷ lệ nhất định của mỗi kết quả khác (Ví dụ: Kiểm tra tỷ lệ Lowe)
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))
		# Tính toán phép đồng dạng (homography) cần ít nhất 4 phép khớp
		if len(matches) > 4:
			# Xây dựng hai tập hợp điểm
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])
			# Tính toán phép đồng dạng (homography) giữa hai tập hợp điểm
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
			# Trả về các kết quả khớp cùng với ma trận đồng dạng và trạng thái của mỗi điểm khớp
			return (matches, H, status)
		# Nếu không, không thể tính toán được sự đồng dạng
		return None

	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# Khởi tạo hình ảnh trực quan đầu ra
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB
		# Lặp lại các matches
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# Chỉ xử lý kết quả khớp nếu keypoint đã được xác định khớp thành công
			if s == 1:
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
		# Trả về hình ảnh trực quan
		return vis