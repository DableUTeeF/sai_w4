import cv2

path = '/media/palm/BiggerData/denso/Denso-Trainingset/20200622/CtlEquip_10/CtlEquip_10_5452_20200622094538.mp4'

cap = cv2.VideoCapture(path)
frames = [None, None, None, None, None]
idx = 0
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    if idx % 15 == 0:
        frames = [*frames[1:], frame]
    idx += 1

for i in range(5):
    cv2.rectangle(frames[i], (490, 58), (630, 190), (0, 255, 0))
    cv2.imshow(str(i), frames[i])
cv2.waitKey()
