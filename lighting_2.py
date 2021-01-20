import numpy as np
from matplotlib import pyplot as plt

def running_mean(x, N):
    return np.convolve(x, np.ones(N)/N, mode='valid')

if __name__ == '__main__':
    a_30 = np.load('/home/palm/PycharmProjects/denso/CtlEquip_10_5540_20200630110744.npz.npy')
    a_30_2 = running_mean(a_30, 60*15)
    a_15 = np.load('/home/palm/PycharmProjects/denso/CtlEquip_10_4859_20200415110033.npz.npy')
    a_15_2 = running_mean(a_15, 60*15)
    for file in [a_15_2, a_30_2]:
        hist_drop_frame = 0
        hist_rise_frame = 0
        for i in range(len(file)-1000):
            hist_sum = file[i]
            if hist_sum > 50:
                tresh = 10
            else:
                tresh = 5
            if np.average(file[i-500:i-450]) - hist_sum > tresh and np.average(file[i-500:i-450]) - np.average(file[i:i+500]) > tresh:
                hist_drop_frame = i
            elif hist_sum - np.average(file[i-500:i-450]) > tresh and np.average(file[i:i+500]) - np.average(file[i-500:i-450]) > tresh:
                hist_rise_frame = i
        print(hist_drop_frame)
        print(hist_rise_frame)
        print(hist_rise_frame - hist_drop_frame > 14*60*14.99)

