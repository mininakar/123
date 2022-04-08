import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import time

#novii komentarii!
#odnako zdravstvuite


# ������� ����� ������ �� � �������
def create_and_train_model(in_data_train, out_data_train):
    # ��� ������� �������� ������, ����� Keras ������� ����� ����� ������ TensorFlow:
    # Keras requires TensorFlow 2.2 or higher. Install TensorFlow via `pip install tensorflow`
    # � ������ ������ ������������ pip install --user --upgrade tensorflow � ������������� ����� ����������
    model = Sequential() # ������ �� - ���� ������� ���������������
    model.add(Input(shape=(in_data_train.shape[1],))) # ������� ����
    # ��������� 2 ������������ (Dense) ���� � ������������� �������� ���������
    model.add(Dense(5, activation="sigmoid")) # ������������� ���� (5 ��������)
    model.add(Dense(1, activation="sigmoid")) # �������� ���� (1 ������)

    # ����������� ������ � ������������� ��������� ������������ ����� (��������� ��������)
    model.compile(loss='mean_absolute_error', optimizer=tf.optimizers.RMSprop(learning_rate=0.005))

    start_time = time.time() # ������� ����� (����� ��������)
    # ��������: epochs - ���������� ���� (������), batch_size - ������ ������ ��������� ������ ��� �������� �����
    history = model.fit(in_data_train, out_data_train, epochs=1000, batch_size=32)
    # ����� ������������ �� �������� �������
    print("--- %s seconds ---" % (time.time() - start_time))
    # ����� ��������� ������
    print("Input size: ", in_data_train.shape[1])
    model.summary()

    return model

# ��������� ��������� ������ � ���� model.json, � ������� ������������ � weights.h5
def save_model(model):
    json_file = 'model.json'
    model_json = model.to_json()

    with open(json_file, 'w') as f:
        f.write(model_json)

    model.save_weights('weights.h5')

# ��������� ������ �� ����� json_file, � ���� �� weights.h5
def load_model(json_file, weights_file):
    with open(json_file, 'r') as f:
        loaded_model = model_from_json(f.read())

    loaded_model.load_weights(weights_file)

    return loaded_model

if __name__ == '__main__':
    dataset = pd.read_csv("dataset.csv", sep=',', header='infer', names=None, encoding="utf-8")  # ���������� Pandas DataFrame
    pd.set_option("display.max_rows", None, "display.max_columns", None)  # ��������� ������ �������
    print(dataset.shape)  # ������� ����������� ������
    # print(dataset.head(10)) # ������� ������ 10 ����� DataFrame (���������, ��� ���� ���������� �����)

    in_data = dataset.iloc[:,1:10]  # .values # �������� ��� ������ c 2 �� 10 ������� � �������������� � np-������
    out_data = dataset.iloc[:, 10:11].values  # �������� ��� ������ (��������� �������) � �������������� � np-������

    # ������������ ������ - �������� ������������ Xnorm = (Xi - Xmin)/(Xmax - Xmin)) -> [0...1]
    norm = MinMaxScaler()       # ������������ ��� ������� ������
    norm_out = MinMaxScaler()   # ������������ ��� �������� ������
    out_data = norm_out.fit_transform(out_data)     # ����������� �������� ��������

    # One-hot (��������) ����������� - ���������� ������ � �������� �������, ��������:
    # "�������" - 1 0 0
    # "�������" - 0 1 0
    # "�����" - 0 0 1
    # �������� ������� ��� one-hot �����������
    one_hot_cols = ['�����', '��� ����������']
    # ���������� ���������� � Pandas ����� one-hot ����������� get_dummies
    for col_name in one_hot_cols:
        one_hot = pd.get_dummies(in_data[col_name])
        in_data = in_data.drop(col_name, axis=1)
        in_data = in_data.join(one_hot)

    # �������������� ������� � True � False ���������� � 1 � 0 ��������������
    bin_cols = ['������/��������� ����', '������� ��������']
    for col_name in bin_cols:
        in_data[col_name] = in_data[col_name].astype(int)

    # ������� ������ 10 ����� ��������������� ������ ��� ��������
    print(in_data.head(10))

    # ������ MinMax-������������ ������� ��������
    in_data = norm.fit_transform(in_data.values)
    print("����������� ������� ������: ", in_data.shape)

    # ��������� �� ������ ������ ������ ������������� (90%) � �������� ������� (10%)
    in_data_train, in_data_test, out_data_train, out_data_test = train_test_split(in_data, out_data, test_size=0.1)

    # ������� ����� ������ �� � ������� �� ������������� �������
    model = create_and_train_model(in_data_train, out_data_train)
    # ���� ���������� ��������� ����������� ������ �� �����, ���������������� ������ ���� � ��������������� ������ ����
    #model = load_model('model.json', 'weights.h5')
    # ��������� ����� ��������� ������ �������� ����� ������ ��� ������ ��������
    out_pred = model.predict(in_data_test)

    # ���������� (out_pred) � ��������������� �� ��������� �������� (out_data_test) �������� � ����������� ������
    predicted = list()
    for i in range(len(out_pred)):
        predicted.append(out_pred[i][0])

    test = list()
    for i in range(len(out_data_test)):
        test.append(out_data_test[i][0])
    # ��������� ������������� ������
    approx_err = mean_absolute_percentage_error(predicted, test)
    print('Approximation error:', approx_err*100)

    # ���������������� ������ ����, ���� ������ ������� �������� �������� ����,
    # ����������� � �������� ��������� (����������� �������� �������� ������������)
    # print(norm_out.inverse_transform(out_data_test))

    print("Save model to file ?:")
    q = input()
    if q.lower() == 'y':
        save_model(model)



