
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    y_pred_val[:, -1] = np.clip(y_pred_val[:, -1], 0, 1)  # NH3_remove_eff 값 클리핑
    val_mse = mean_squared_error(y_val, y_pred_val)
    
    y_pred_test = model.predict(X_test)
    y_pred_test[:, -1] = np.clip(y_pred_test[:, -1], 0, 1)  # NH3_remove_eff 값 클리핑
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    return val_mse, test_mse

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim-1)
        self.fc4 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out1 = self.fc3(x)
        out2 = self.sigmoid(self.fc4(x))
        return torch.cat((out1, out2), dim=1)

def train_model(model, criterion, optimizer, train_loader, val_loader, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}')

def evaluate_pytorch_model(model, criterion, data_loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(data_loader)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork(input_dim, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 모델 학습
    train_model(model, criterion, optimizer, train_loader, val_loader, args.epochs)

    # 신경망 평가
    nn_val_mse = evaluate_pytorch_model(model, criterion, val_loader)
    nn_test_mse = evaluate_pytorch_model(model, criterion, test_loader)

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

    main(args)
