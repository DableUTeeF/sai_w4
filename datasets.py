from torch.utils.data import Dataset
import pandas as pd


class DensoDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass


def getdata(csv):
    csv = pd.read_csv('/media/palm/BiggerData/denso/Denso-Trainingset/2020-04-16.csv')
    csv = csv[csv['has_actual_output'] != 'Corrupted Video']
    has_actual_output = csv['has_actual_output'].values
    ct_include_breaktime = csv['ct_include_breaktime'].values
    n_ct = csv['n_ct'].values
    y = [has_actual_output, ct_include_breaktime, n_ct]  # todo: don't use n_ct for the time being

    datetime = csv['d_datetime'].values
    equipment_control = csv['s_equipment_control'].values



if __name__ == '__main__':
    l = """/media/palm/BiggerData/denso/Denso-Trainingset/2020-04-16.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-04-17.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-04-20.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-04-21.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-04-22.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-06-01.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-06-02.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-06-22.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-06-23.csv
/media/palm/BiggerData/denso/Denso-Trainingset/2020-06-29.csv""".split('\n')

    getdata('/media/palm/BiggerData/denso/Denso-Trainingset/2020-04-16.csv')
