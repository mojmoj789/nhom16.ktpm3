import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Chuẩn bị dữ liệu
file_path = r'C:\Users\Admin\Documents\GitHub\nhom16.ktpm3\dtset.csv'
data = pd.read_csv(file_path)

# Kiểm tra dữ liệu thiếu và thay thế nếu cần thiết
data.fillna(data.median(), inplace=True)

# Định nghĩa biến đầu vào (X) và biến mục tiêu (y)
X = data.drop('PRICE', axis=1)
y = data['PRICE']

# 2. Chia tách dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Huấn luyện mô hình Ridge Regression
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# 5. Dự đoán và đánh giá mô hình
y_pred = model.predict(X_test_scaled)

# Tính toán độ lỗi trung bình bình phương (MSE) và hệ số xác định (R^2)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# In ra kết quả đánh giá mô hình
print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

# 6. Trực quan hóa kết quả dự đoán
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices (Ridge Regression)')
plt.grid(True)
plt.show()