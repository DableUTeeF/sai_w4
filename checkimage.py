import os
import cv2

cv2.drawContours
if __name__ == '__main__':
    root = '/media/palm/BiggerData/denso/images'
    for file in os.listdir(root):
        image = cv2.imread(os.path.join(root, file))
        image = cv2.rectangle(image, (490, 58), (630, 190), (0, 255, 0))
        cv2.imwrite(os.path.join(root, 'test', file), image)
