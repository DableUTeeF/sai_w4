import cv2
import numpy as np

if __name__ == '__main__':
    path = '/media/palm/BiggerData/denso/Denso-Trainingset/20200420/CtlEquip_10/CtlEquip_10_4917_20200420094623.mp4'
    cap = cv2.VideoCapture(path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    old_gray[:58, :] = 0
    old_gray[190:, :] = 0
    old_gray[:, :520] = 0
    old_gray[:, 660:] = 0

    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    backSub = cv2.createBackgroundSubtractorMOG2()

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    for i in range(frame_count):
        ret, frame = cap.read()
        if i < frame_count - 500:
            continue
        if frame is None:
            break

        frame[:30, :432, :] = 0
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)
        # frame_gray[58:190, 490:630] = 0
        frame_gray[:58, :] = 0
        frame_gray[190:, :] = 0
        frame_gray[:, :520] = 0
        frame_gray[:, 660:] = 0
        # frame_gray = backSub.apply(frame_gray)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)

        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cv2.destroyAllWindows()
    cap.release()


