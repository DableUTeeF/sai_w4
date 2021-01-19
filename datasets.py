from torch.utils.data import Dataset
import pandas as pd
import os
from datetime import datetime
import cv2
import json
import torch
from PIL import Image
from torchvision import transforms
import numpy as np


class DensoDataset(Dataset):
    def __init__(self, imgpath='/media/palm/BiggerData/denso/images', imsize=224, transform=None, sequence=True):
        self.data = json.load(open('/media/palm/BiggerData/denso/annotations/data.json'))
        self.imgpath = imgpath
        self.imsize = imsize
        self.transforms = transform
        self.sequence = sequence
        if self.transforms is None:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.transforms = transforms.Compose([transforms.Resize(imsize),
                                                  transforms.ToTensor(),
                                                  normalize,
                                                  ])

    def __len__(self):
        if self.sequence:
            return len(self.data)
        else:
            return len(self.data) * 10

    def __getitem__(self, index):
        """{"include_breaktime": False, "has_actual_output": True, "prefix": "20200416_20200416081455"}"""
        if self.sequence:
            data = self.data[index]
            if self.sequence:
                x = torch.zeros((10, 3, self.imsize, self.imsize))
                for i in range(10):
                    image = Image.open(os.path.join(self.imgpath, f"{data['prefix']}_{i}.png"))
                    image = self.transforms(image)
                    x[i] = image
            return x, int(data['has_actual_output'])
        else:
            idx, i = divmod(index, 10)
            data = self.data[idx]
            image = Image.open(os.path.join(self.imgpath, f"{data['prefix']}_{i}.png"))
            image = self.transforms(image)
            return image, int(data['has_actual_output'])


def create_image():
    csvs = """/media/palm/BiggerData/denso/Denso-Trainingset/2020-04-16.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-04-17.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-04-20.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-04-21.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-04-22.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-06-01.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-06-02.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-06-22.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-06-23.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-06-29.csv""".split('\n')
    folders = """/media/palm/BiggerData/denso/Denso-Trainingset/20200416
/media/palm/BiggerData/denso/Denso-Trainingset/20200417
/media/palm/BiggerData/denso/Denso-Trainingset/20200420
/media/palm/BiggerData/denso/Denso-Trainingset/20200421
/media/palm/BiggerData/denso/Denso-Trainingset/20200422
/media/palm/BiggerData/denso/Denso-Trainingset/20200601
/media/palm/BiggerData/denso/Denso-Trainingset/20200602
/media/palm/BiggerData/denso/Denso-Trainingset/20200622
/media/palm/BiggerData/denso/Denso-Trainingset/20200623
/media/palm/BiggerData/denso/Denso-Trainingset/20200629""".split('\n')
#     csvs = ['/media/palm/BiggerData/denso/Denso-Trainingset/2020-04-16.csv']
#     folders = ['/media/palm/BiggerData/denso/Denso-Trainingset/20200416']
    data = []
    for folder, csv in zip(folders, csvs):
        vids = os.listdir(os.path.join(folder, 'CtlEquip_10'))
        vids = sorted(vids)
        vidname = []
        for idx, vid in enumerate(vids):
            timestamp = vid.split('_')[-1].replace('.mp4', '')
            timestamp = datetime.strptime(timestamp, '%Y%m%d%H%M%S')
            vidname.append((timestamp, vid))
        print(vidname)
        csv = pd.read_csv(csv)
        csv = csv[csv['has_actual_output'] != 'Corrupted Video']
        has_actual_outputs = csv['has_actual_output'].values
        ct_include_breaktimes = csv['ct_include_breaktime'].values
        d_datetimes = csv['d_datetime'].values
        for has_actual_output, ct_include_breaktime, d_datetime in zip(has_actual_outputs, ct_include_breaktimes, d_datetimes):
            d_datetime = datetime.strptime(d_datetime, '%Y-%m-%d %H:%M:%S')
            found_something = 0
            try:
                for i in range(len(vidname) - 1):
                    if vidname[i][0] < d_datetime <= vidname[i + 1][0]:
                        found_something = 1
                        frame_pos = (d_datetime - vidname[i][0]).seconds * 15
                        cap = cv2.VideoCapture(os.path.join(folder, 'CtlEquip_10', vidname[i][1]))
                        idx = 0
                        frames = [None for _ in range(10)]
                        while True:
                            ret, frame = cap.read()
                            if frame is None:
                                break
                            if idx % 5 == 0:
                                frames = [*frames[1:], frame]
                            idx += 1
                        break
                    elif vidname[0][0] >= d_datetime:  # the first vids of the days
                        found_something = 2
                        frame_pos = (d_datetime - vidname[i][0]).seconds * 15
                        cap = cv2.VideoCapture(os.path.join(folder, 'CtlEquip_10', vidname[i][1]))
                        idx = 0
                        frames = []
                        while True:
                            ret, frame = cap.read()
                            if len(frames) > 10:
                                break
                            if idx % 5 == 0 and idx > 1:
                                frames.append(frame)
                            idx += 1
                        break
            except:
                print('error', d_datetime)
            if found_something > 0:
                try:
                    print('top' if found_something == 1 else 'bot', end=' ')
                    print(d_datetime, vidname[i][1])
                    for j, frame in enumerate(frames):
                        cv2.imwrite(os.path.join('/media/palm/BiggerData/denso/images',
                                                 f"{vidname[i][1].split('_')[-1].replace('.mp4', '')}_{j}.png"),
                                    frame)
                    obj = {'include_breaktime': ct_include_breaktime.lower() == 'yes',
                           'has_actual_output': has_actual_output.lower() == 'yes',
                           'prefix': f"{vidname[i][1].split('_')[-1].replace('.mp4', '')}"}
                    data.append(obj)
                except Exception as e:
                    print(e)
            else:
                print(d_datetime)
    json.dump(data, open('/media/palm/BiggerData/denso/annotations/data.json', 'w'))


def check_csv():
    news = """/media/palm/BiggerData/denso/super-ai-engineer-denso-lasi/train_csv/2020-04-16.csv
/media/palm/BiggerData/denso/super-ai-engineer-denso-lasi/train_csv/2020-04-17.csv
/media/palm/BiggerData/denso/super-ai-engineer-denso-lasi/train_csv/2020-04-20.csv
/media/palm/BiggerData/denso/super-ai-engineer-denso-lasi/train_csv/2020-04-21.csv
/media/palm/BiggerData/denso/super-ai-engineer-denso-lasi/train_csv/2020-04-22.csv
/media/palm/BiggerData/denso/super-ai-engineer-denso-lasi/train_csv/2020-06-01.csv
/media/palm/BiggerData/denso/super-ai-engineer-denso-lasi/train_csv/2020-06-02.csv
/media/palm/BiggerData/denso/super-ai-engineer-denso-lasi/train_csv/2020-06-22.csv
/media/palm/BiggerData/denso/super-ai-engineer-denso-lasi/train_csv/2020-06-23.csv
/media/palm/BiggerData/denso/super-ai-engineer-denso-lasi/train_csv/2020-06-29.csv""".split('\n')
    olds = """/media/palm/BiggerData/denso/Denso-Trainingset/2020-04-16.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-04-17.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-04-20.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-04-21.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-04-22.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-06-01.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-06-02.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-06-22.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-06-23.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-06-29.csv""".split('\n')
    for new, old in zip(news, olds):
        new = pd.read_csv(new)
        old = pd.read_csv(old)
        print(np.sum((new != old).values))



if __name__ == '__main__':
    check_csv()
