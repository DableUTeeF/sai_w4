import os
import pandas as pd
from datetime import datetime
import json


if __name__ == '__main__':
    vids = os.listdir('/media/palm/BiggerData/denso/testdata/all')
    vids = sorted(vids)
    vidname = []
    for idx, vid in enumerate(vids):
        timestamp = vid.split('_')[-1].replace('.mp4', '')
        timestamp = datetime.strptime(timestamp, '%Y%m%d%H%M%S')
        vidname.append((timestamp, vid))
    print(vidname)

    csv = pd.read_csv('/media/palm/BiggerData/denso/super-ai-engineer-denso-lasi/test.csv')
    d_datetimes = csv['d_datetime'].values
    ids = csv['Ids'].values
    data = []
    for idx, d_datetime in zip(ids, d_datetimes):
        d_datetime = datetime.strptime(d_datetime, '%Y-%m-%d %H:%M:%S')
        found_something = False
        for i in range(len(vidname)-1):
            if vidname[i][0] < d_datetime <= vidname[i+1][0]:
                found_something = True
                frame_pos = (d_datetime - vidname[i][0]).seconds * 15
                obj = {'frame': frame_pos,
                       'filename': vidname[i][1],
                       'loc': 'end',
                       'date': d_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                       'id': int(idx)}
                data.append(obj)
                print(obj)
                break
            elif vidname[0][0] >= d_datetime:
                found_something = True
                frame_pos = (d_datetime - vidname[i][0]).seconds * 15
                obj = {'frame': frame_pos,
                       'filename': vidname[i][1],
                       'loc': 'start',
                       'date': d_datetime}
                data.append(obj)
                print(obj)
                break

        if not found_something:
            if vidname[-1][0] < d_datetime:
                found_something = True
                frame_pos = (d_datetime - vidname[-1][0]).seconds * 15
                obj = {'frame': frame_pos,
                       'filename': vidname[-1][1],
                       'loc': 'end',
                       'date': d_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                       'id': int(idx)}
                data.append(obj)
                print(obj)
                break
            print(d_datetime)
    json.dump(data, open('test.json', 'w'))
