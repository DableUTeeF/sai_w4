from bg_subtraction import process_vids
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
    answer = []
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
                result = process_vids(vidname[i][1], '/media/palm/BiggerData/denso/testdata/all')
                obj = {'frame': frame_pos,
                       'filename': vidname[i][1],
                       'loc': 'end',
                       'date': d_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                       'id': int(idx),
                       'result': result,
                       }
                print(obj)
                answer.append(obj)
                break
            elif vidname[0][0] >= d_datetime:
                found_something = True
                frame_pos = (d_datetime - vidname[i][0]).seconds * 15
                obj = {'frame': frame_pos,
                       'filename': vidname[i][1],
                       'loc': 'start',
                       'date': d_datetime.strftime('%Y-%m-%d %H:%M:%S')}

                print(obj)
                break

        if not found_something:
            if vidname[-1][0] < d_datetime:
                found_something = True
                frame_pos = (d_datetime - vidname[-1][0]).seconds * 15
                result = process_vids(vidname[-1][1], '/media/palm/BiggerData/denso/testdata/all')
                obj = {'frame': frame_pos,
                       'filename': vidname[-1][1],
                       'loc': 'end',
                       'date': d_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                       'id': int(idx),
                       'result': result}
                answer.append(obj)
                print(obj)
                break
            print(d_datetime)

    with open('submission.csv', 'w') as wr:
        wr.write('Ids,Prediction\n')
        for obj in answer:
            wr.write(f"{obj['id']},")
            if obj['result'] > 100:
                wr.write('"Yes')
            else:
                wr.write("No")
            wr.write(',No"\n')
