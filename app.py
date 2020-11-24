from flask import Flask, url_for, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import time
import pymysql
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

app = Flask(__name__)

# 配置参数
class Config(object):
    """配置参数"""
    # sqlalchemy的配置参数
    SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root:root@127.0.0.1:3306/wholedata"
    # 设置sqlalchemy自动跟踪数据库
    SQLALCHEMY_TRACK_MODIFICATIONS = True


app.config.from_object(Config)
db = SQLAlchemy(app)


class EleData(db.Model):
    __tablename__= 'eledata'
    id = db.Column(db.Integer, primary_key=True)  # 主键 整型的主键默认设置为自增
    date = db.Column(db.String(30), nullable=True)
    electricity = db.Column(db.FLOAT, nullable=True)
    preElectricity = db.Column(db.FLOAT, nullable=True)
    usefulWork = db.Column(db.FLOAT, nullable=True)
    preUsefulWork = db.Column(db.FLOAT, nullable=True)
    uselessWork = db.Column(db.FLOAT, nullable=True)
    preUselessWork = db.Column(db.FLOAT, nullable=True)


class DealData:
    @classmethod
    def _create_dataset(cls, dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    @classmethod
    def _read_data(cls, data_dir, model_dir, y_index):
        data = pd.read_csv(data_dir)
        data['Time'] = pd.to_datetime(data['Time'])
        data = data.sort_values("Time")
        new_data = pd.DataFrame(index=range(0, len(data)), columns=['Date', 'y'])
        new_data['Date'] = data['Time'].values
        new_data['y'] = data[y_index].values
        new_data.set_index('Date', drop=True, inplace=True)

        dataset = new_data.values
        dataset = dataset[30000:]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        scaled_data = scaled_data[25000:]

        x, y = cls._create_dataset(scaled_data, look_back=10)

        x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
        model = joblib.load(model_dir)
        y_predict = model.predict(x)

        y_predict = scaler.inverse_transform(y_predict)
        y = scaler.inverse_transform([y])

        dataset = dataset[25000:]
        dataset = dataset.flatten()
        return dataset, y_predict, y


def writedata():
        model_I_dir = r'D:\pycharmPro\stationForewarning\5_LSTM\save\lstm_current2.model'
        model_Y_dir = r'D:\pycharmPro\stationForewarning\5_LSTM\save\lstm_usefulwork.model'
        model_W_dir = r'D:\pycharmPro\stationForewarning\5_LSTM\save\lstm_uselesswork.model'
        data_dir = r'D:\SJTU\机器学习\stationData\给的数据\自动化数据\2019.8.9—9.19.csv'
        dataset_I, predict_I, I = DealData._read_data(data_dir, model_I_dir, 'I')
        dataset_Y, predict_Y, Y = DealData._read_data(data_dir, model_Y_dir, 'Y')
        dataset_W, predict_W, W = DealData._read_data(data_dir, model_W_dir, 'W')
        print(np.float(dataset_I[0]))
        print(type(dataset_I[0]))
        print(predict_I[0][0])
        print(dataset_Y[0])
        print(predict_Y[0][0])
        print(dataset_W[0])
        print(predict_W[0][0])

        for i in range(len(predict_I)):
            eledata = EleData()
            eledata.electricity = np.float(dataset_I[i])
            eledata.preElectricity = np.float(predict_I[i][0])
            eledata.usefulWork = np.float(dataset_Y[i])
            eledata.preUsefulWork = np.float(predict_Y[i][0])
            eledata.uselessWork = np.float(dataset_W[i])
            eledata.preUselessWork = np.float(predict_W[i][0])
            db.session.add(eledata)
            db.session.commit()
            time.sleep(1)

writedata()



@app.route('/ele_data', methods=['GET', "POST"])
def ele_data():
    if request.method == 'POST':
        print(request.json)
        return jsonify('Hello World!')


if __name__ == '__main__':
    app.run()
