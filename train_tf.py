## python your_script.py --data_path /path/to/your/data.csv --rf_estimators 150 --gb_estimators 150 --gb_learning_rate 0.05 --epochs 50 --batch_size 16
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Concatenate

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    y_pred_val[:, -1] = np.clip(y_pred_val[:, -1], 0, 1)  # NH3_remove_eff 값 클리핑
    val_mse = mean_squared_error(y_val, y_pred_val)
    
    y_pred_test = model.predict(X_test)
    y_pred_test[:, -1] = np.clip(y_pred_test[:, -1], 0, 1)  # NH3_remove_eff 값 클리핑
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    return val_mse, test_mse

def main(args):
    # 데이터 로드
    data = pd.read_csv(args.data_path)

    # 입력 변수와 출력 변수를 분리
    X = data[['conversion', 'gas_pressure', 'gas_temp', 'column_pressure', 'water_flowrate', 'tray_number']]
    y = data[['liqout_flowrate_water', 'liqout_flowrate_hydrogen', 'liqout_flowrate_nitrogen', 
              'liqout_flowrate_ammonia', 'liqout_temp', 'liqout_pressure', 
              'liqout_flowrate_total', 'NH3_remove_eff']]

    # 학습, 검증, 테스트 세트로 분할
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    # 데이터 표준화
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)

    # 선형 회귀
    linear_model = LinearRegression()
    linear_val_mse, linear_test_mse = evaluate_model(linear_model, X_train, y_train, X_val, y_val, X_test, y_test)
    print(f'Linear Regression - Validation MSE: {linear_val_mse}, Test MSE: {linear_test_mse}')

    # 랜덤 포레스트
    rf_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=args.rf_estimators, random_state=42))
    rf_val_mse, rf_test_mse = evaluate_model(rf_model, X_train, y_train, X_val, y_val, X_test, y_test)
    print(f'Random Forest - Validation MSE: {rf_val_mse}, Test MSE: {rf_test_mse}')

    # 그래디언트 부스팅
    gb_model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=args.gb_estimators, learning_rate=args.gb_learning_rate, random_state=42))
    gb_val_mse, gb_test_mse = evaluate_model(gb_model, X_train, y_train, X_val, y_val, X_test, y_test)
    print(f'Gradient Boosting - Validation MSE: {gb_val_mse}, Test MSE: {gb_test_mse}')

    # 신경망 모델 정의
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    nn_model = Sequential()
    nn_model.add(Dense(64, input_dim=input_dim, activation='relu'))
    nn_model.add(Dense(64, activation='relu'))

    # 전체 출력 레이어를 분리해서, NH3_remove_eff 에 sigmoid 를 적용
    output_layers = [Dense(output_dim-1)(nn_model.layers[-1].output), Dense(1, activation='sigmoid')(nn_model.layers[-1].output)]
    output = Concatenate()(output_layers)

    nn_model = tf.keras.Model(inputs=nn_model.input, outputs=output)

    # 모델 컴파일
    nn_model.compile(optimizer='adam', loss='mean_squared_error')

    # 모델 학습
    nn_model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(X_val, y_val))

    # 신경망 평가
    y_pred_val = nn_model.predict(X_val)
    nn_val_mse = mean_squared_error(y_val, y_pred_val)

    y_pred_test = nn_model.predict(X_test)
    nn_test_mse = mean_squared_error(y_test, y_pred_test)

    print(f'Neural Network - Validation MSE: {nn_val_mse}, Test MSE: {nn_test_mse}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--rf_estimators', type=int, default=100, help='Number of estimators for Random Forest')
    parser.add_argument('--gb_estimators', type=int, default=100, help='Number of estimators for Gradient Boosting')
    parser.add_argument('--gb_learning_rate', type=float, default=0.1, help='Learning rate for Gradient Boosting')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for Neural Network')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for Neural Network')
    args = parser.parse_args()

    # GPU 설정
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    main(args)
