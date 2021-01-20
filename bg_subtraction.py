import cv2
import numpy as np


if __name__ == '__main__':
    file = '/media/palm/BiggerData/denso/Denso-Trainingset/20200420/CtlEquip_10/CtlEquip_10_4917_20200420094623.mp4'
    cap = cv2.VideoCapture(file)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps

    print('fps', fps)
    print('frame_count', frame_count)

    backSub = cv2.createBackgroundSubtractorMOG2()
    subs = [None for i in range(100)]
    frames = [None for i in range(100)]
    for i in range(frame_count):
        ret, raw_frame = cap.read()
        if i < frame_count - 500:
            continue
        if raw_frame is None:
            break
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.putText(frame, f'{i}', (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,))
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        sub = backSub.apply(frame)

        # sub = cv2.erode(sub, np.ones((2, 2)))

        # if i > frame_count - 480:
        #     avg = np.average(frames, 0).astype('uint8')
        #     diff = cv2.absdiff(frames[-1], frame)
        #     _, diff = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
        #     cv2.imshow('diff', diff)
        cv2.imshow('sub', sub)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        contours, h = cv2.findContours(sub, 1, 2)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2000:  # or len(approx) != 4:
                continue
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            x = approx[..., 0]
            y = approx[..., 1]
            xmin = min(x)[0]
            xmax = max(x)[0]
            ymin = min(y)[0]
            ymax = max(y)[0]
            # cv2.drawContours(ct, [cnt], 0, 255, 2)
            sub = cv2.rectangle(sub,
                                (xmin, ymin),
                                (xmax, ymax), (255,), 2)
        raw_frame = cv2.rectangle(raw_frame, (525, 50), (545, 70), (0, 255, 0))
        cv2.imshow('raw_frame', raw_frame)
        frames = [*frames[1:], raw_frame]
        subs = [*subs[1:], sub]
print(np.sum(sub[50:70, 525:545]==255))
cv2.waitKey()
for i in range(100):
    cv2.imwrite(f'/media/palm/BiggerData/denso/subs/{i}_f.jpg', frames[i])
    cv2.imwrite(f'/media/palm/BiggerData/denso/subs/{i}_s.jpg', subs[i])
# cv2.waitKey()


