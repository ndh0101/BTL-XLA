from panorama import Stitcher
import argparse
import imutils
import cv2

# Xây dựng câu lệnh phân tích đối số và lấy các đối số
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True)
ap.add_argument("-s", "--second", required=True)
args = vars(ap.parse_args())

# Tải hai hình ảnh và thay đổi kích thước của chúng để có chiều rộng là 400 pixel (để xử lý nhanh hơn)
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])
imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)

# Ghép các hình ảnh lại với nhau để tạo thành một bức tranh toàn cảnh
stitcher = Stitcher()
(result, vis, imageA_key, imageB_key) = stitcher.stitch([imageA, imageB], showMatches=True)

# Hiển thị hình ảnh
cv2.imshow("Image A", imageA_key)
cv2.imshow("Image B", imageB_key)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)

# Chạy lệnh trên Terminal: python test.py --first (đường dẫn ảnh 1) --second (đường dẫn ảnh 2)