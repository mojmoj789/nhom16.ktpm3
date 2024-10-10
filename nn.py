import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Chuẩn bị dữ liệu
file_path = r'C:\Users\Admin\Documents\GitHub\nhom16.ktpm3\dtset.csv'
data = pd.read_csv(file_path)

# Kiểm tra dữ liệu thiếu và thay thế nếu cần thiết
data.fillna(data.median(), inplace=True)

# Định nghĩa biến đầu vào (X) và biến mục tiêu (y)
X = data.drop('PRICE', axis=1)
y = data['PRICE']

# Sử dụng một phần nhỏ của dữ liệu để kiểm tra
X = X.sample(frac=0.1, random_state=42)
y = y.sample(frac=0.1, random_state=42)

# 2. Chia tách dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Xây dựng và huấn luyện mô hình Neural Network sử dụng MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Dự đoán và đánh giá mô hình
y_pred = model.predict(X_test_scaled)

# Tính toán độ lỗi trung bình bình phương (MSE), lỗi tuyệt đối trung bình (MAE) và hệ số xác định (R^2)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# In ra kết quả đánh giá mô hình
print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'R^2: {r2}')

# 6. Sử dụng matplotlib để vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Prices')
plt.show()