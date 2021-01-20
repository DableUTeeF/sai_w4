import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

def running_mean(x, N):
    return np.convolve(x, np.ones(N)/N, mode='valid')


if __name__ == '__main__':
    # path = '/media/palm/BiggerData/denso/Denso-Trainingset/20200416/CtlEquip_10/CtlEquip_10_4887_20200416091828.mp4'
    # path = '/media/palm/BiggerData/denso/Denso-Trainingset/20200417/CtlEquip_10/CtlEquip_10_4898_20200417090131.mp4'  # 22660-22661
    path = '/media/palm/BiggerData/denso/testdata/all/CtlEquip_10_4859_20200415110033.mp4'
    # path = '/media/palm/BiggerData/denso/testdata/all/CtlEquip_10_5540_20200630110744.mp4'
    out = '/media/palm/BiggerData/denso/testdata/hist'
    cap = cv2.VideoCapture(path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print('fps', fps)
    print('frame_count', frame_count)

    hist_drop_frame = 0
    hist_rise_frame = 0
    previous_hist_sum = None
    stats = []
    for i in range(frame_count):
        ret, raw_frame = cap.read()
        if raw_frame is None:
            continue
        # if not start_frame <= i <= end_frame:
        #     continue
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)[260:, :220]
        histogram = cv2.calcHist([frame], [0], None, [64], [0, 255])
        hist_max = np.argsort(histogram, 0)[::-1]
        hist_sum = np.sum(hist_max[:5])
        if previous_hist_sum is not None:
            if previous_hist_sum / hist_sum > 1.5:
                hist_drop_frame = i
            elif hist_sum / previous_hist_sum > 1.5 and hist_drop_frame > 0:
                hist_rise_frame = i
                if 14*60*fps < hist_rise_frame - hist_drop_frame:
                    raise ValueError(f"Found break at {hist_drop_frame} to {hist_rise_frame}")
        stats.append(hist_sum)
        previous_hist_sum = hist_sum

        # cv2.imwrite(os.path.join(out, f'{i:07d}.png'),
        #             raw_frame)
    print(hist_drop_frame)
    print(hist_rise_frame)
    np.save('CtlEquip_10_4859_20200415110033.npz', stats)
    plt.plot(running_mean(np.array(stats), int(fps*60)))
    plt.show()
