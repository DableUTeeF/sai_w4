import cv2
import numpy as np
import os
import multiprocessing
from functools import partial


def process_vids(file, root):
    cap = cv2.VideoCapture(os.path.join(root, file))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    file = file.split('_')[-1].replace('.mp4', '')
    if fps == 0:
        return
    if frame_count == 0:
        return
    if os.path.exists(os.path.join('/media/palm/BiggerData/denso/subs', file + f'_f.jpg')):
        return
    # print('fps', fps)
    # print('frame_count', frame_count)

    backSub = cv2.createBackgroundSubtractorMOG2()
    subs = [None for i in range(150)]
    frames = [None for i in range(150)]
    sub = None
    for i in range(frame_count):
        ret, raw_frame = cap.read()
        if i < frame_count - 150 or i < 75:
            continue
        if raw_frame is None:
            break
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        sub = backSub.apply(frame)

        # cv2.imshow('sub', sub)
        # k = cv2.waitKey(1) & 0xff
        # if k == 27:
        #     raise KeyboardInterrupt
        sub = cv2.rectangle(sub, (533, 48), (557, 72), (255, 0, 0))
        raw_frame = cv2.rectangle(raw_frame, (535, 50), (555, 70), (0, 255, 0))
        # cv2.imshow('raw_frame', raw_frame)
        frames = [*frames[1:], raw_frame]
        subs = [*subs[1:], sub]
    max_sub = 0
    frame = frames[-1]
    for s, f in zip(subs, frames):
        if not isinstance(s, np.ndarray):
            continue
        if np.sum(s[50:70, 535:555] == 255) > max_sub:
            max_sub = np.sum(s[50:70, 535:555] == 255)
            frame = f
            sub = s
    if sub is None:
        return
    # print(np.sum(sub[50:70, 525:545] == 255))
    # print()
    try:
        cv2.imwrite(os.path.join('/media/palm/BiggerData/denso/subs', file + f'_f.jpg'),
                    frame)
        cv2.imwrite(os.path.join('/media/palm/BiggerData/denso/subs', file + f'_{np.sum(sub[50:70, 525:555] == 255)}.jpg'),
                    sub)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    root = '/media/palm/BiggerData/denso/Denso-Trainingset/'
    for folder in os.listdir(root):
        if not os.path.isdir(os.path.join(root, folder)):
            continue
        files = os.listdir(os.path.join(root, folder, 'CtlEquip_10'))

            # cap = cv2.VideoCapture(os.path.join(root, folder, 'CtlEquip_10', file))
        with multiprocessing.Pool(processes=8) as pool:
            results = pool.map(partial(process_vids, root=os.path.join(root, folder, 'CtlEquip_10')),
                               files)
