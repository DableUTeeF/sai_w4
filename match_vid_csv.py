import os
import pandas as pd
from datetime import datetime

if __name__ == '__main__':
    vids = os.listdir('/media/palm/BiggerData/denso/Denso-Trainingset/20200622/CtlEquip_10')
    vids = sorted(vids)
    vidname = []
    for idx, vid in enumerate(vids):
        timestamp = vid.split('_')[-1].replace('.mp4', '')
        timestamp = datetime.strptime(timestamp, '%Y%m%d%H%M%S')
        vidname.append((timestamp, vid))
    print(vidname)

    csv = pd.read_csv('/media/palm/BiggerData/denso/Denso-Trainingset/2020-06-22.csv')
    has_actual_outputs = csv['has_actual_output'].values
    ct_include_breaktimes = csv['ct_include_breaktime'].values
    d_datetimes = csv['d_datetime'].values
    data = []
    for has_actual_output, ct_include_breaktime, d_datetime in zip(has_actual_outputs, ct_include_breaktimes, d_datetimes):
        d_datetime = datetime.strptime(d_datetime, '%Y-%m-%d %H:%M:%S')
        for i in range(len(vidname)-1):
            if vidname[i][0] < d_datetime < vidname[i+1][0]:
                frame_pos = (d_datetime - vidname[i][0]).seconds * 15
                obj = {'frame': frame_pos,
                       'filename': vidname[i][1]}
                data.append(obj)
                print(obj)
