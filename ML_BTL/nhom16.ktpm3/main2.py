import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score

# Đọc dữ liệu từ file
df = pd.read_csv('/ML_BTL/dataset4.csv')

# Chia dữ liệu thành features (biến độc lập) và target (biến phụ thuộc)
X = df.drop(columns=['PRICE'])  # Các đặc trưng
y = df['PRICE']  # Biến mục tiêu là giá nhà

# Kiểm tra và xử lý giá trị thiếu (nếu có)
if df.isnull().sum().any():
    X.fillna(X.mean(), inplace=True)

# Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu (giúp tăng hiệu quả với dữ liệu có giá trị lớn)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Giảm chiều với PCA (giữ lại 95% thông tin)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Base models: Linear Regression, Ridge, MLPRegressor, RandomForest, GradientBoosting
base_models = [
    ('linear', LinearRegression()),
    ('ridge', Ridge(alpha=10.0)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(30, 30), max_iter=2000, learning_rate_init=0.001, activation='relu', random_state=42)),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gbr', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
]

# Final model: Ridge Regression
final_model = Ridge(alpha=50.0)

# Stacking Regressor
stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=final_model)

# Sử dụng GridSearchCV để tối ưu hóa siêu tham số của RandomForestRegressor
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=5, scoring='r2')
grid_search_rf.fit(X_train_pca, y_train)

best_rf_model = grid_search_rf.best_estimator_

# Huấn luyện mô hình Stacking với mô hình tốt nhất từ RandomForest
stacking_regressor.fit(X_train_pca, y_train)

# Dự đoán trên tập kiểm tra
y_pred = stacking_regressor.predict(X_test_pca)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

# Tạo một DataFrame để theo dõi quá trình đánh giá mô hình
evaluation_df = pd.DataFrame(columns=['Model', 'MSE', 'MAE', 'RMSE', 'R2_Score'])

# Tính toán các chỉ số đánh giá
def calculate_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Thêm kết quả vào DataFrame
    evaluation_df.loc[len(evaluation_df)] = [model_name, mse, mae, rmse, r2]

# Tính toán và lưu kết quả cho mô hình Stacking Regressor
calculate_metrics(y_test, y_pred, 'Stacking Regressor')

# Lưu mô hình vào file
with open('best_model.pkl', 'wb') as file:
    pickle.dump(stacking_regressor, file)

# Vẽ biểu đồ giá thực tế so với giá dự đoán
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Giá thực tế', color='blue', marker='o')
plt.plot(y_pred, label='Giá dự đoán', color='red', marker='x')
plt.title('Giá thực tế vs Giá dự đoán')
plt.xlabel('Mẫu')
plt.ylabel('Giá nhà')
plt.legend()
plt.show()

# Xuất DataFrame theo dõi quá trình đánh giá ra file CSV
evaluation_df.to_csv('DataFrame.csv', index=False)
