#2. 使用RNN实现一个天气预测模型，能预测1天和连续5天的最高气温。要求使用tensorboard，提交代码及run目录和可视化截图。
#   数据集：URL_ADDRESS   数据集：https://www.kaggle.com/datasets/smid80/weatherww2
#
import csv

import numpy as np
import torch
from torch import nn


#
#STA、 Date, Precip ,WindGustSpd,MaxTemp,MinTemp,MeanTemp,Snowfall,PoorWeather,YR,MO,DA,PRCP,DR,SPD,MAX,MIN,MEA,SNF,SND,FT,FB,FTI,ITH,PGT,TSHDSBRSGF,SD3,RHX,RHN,RVG,WTE
#站号、日期、  降水量。  最大风 ， 最大温度，最小温度，平均温度 ， 降雪，
def load_weather_data(path):
    res  = {}
    data = []
    with open(path,'r' , encoding = 'utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            res.setdefault(row['STA'],{})[row['Date']] = float(row['MaxTemp'])
    return res


class RNNModel(nn.Module):

    def __init__(self, input_size, hidden_size,num_layers, output_size , model_name):
        drop = 0.3

        super(RNNModel, self).__init__()
        if model_name == 'LSTM':
            self.rnn = nn.LSTM(input_size = input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True , dropout=drop)
            self.fc = nn.Linear(hidden_size, output_size)
        elif model_name == 'GRU':
            self.rnn = nn.GRU(input_size = input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True ,dropout=drop)
            self.fc = nn.Linear(hidden_size, output_size)
        else : # 'RNN'
            self.rnn = nn.RNN( input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True ,dropout=drop)
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output,end = self.rnn(x)
        y = self.fc(output[:, -1, :])
        # y = self.fc(end[-1,:,:])
        return y

if __name__ == '__main__':
    # b t feature
    #
    res = load_weather_data('../data/SummaryofWeather.csv')
    sta_num = len(res) # 159个站点
    print(list(res.values()))
    date_num = len(res[list(res.keys())])
    print(sta_num,date_num)
    # sta , data , temp
    data = torch.randn([sta_num,date_num,1],dtype=torch.float)
    print(data.shape)

    for i , sta in enumerate(res):
        for ii ,date_step  in enumerate( res[sta]):
            data[i,ii,0] = res[sta][date_step]




