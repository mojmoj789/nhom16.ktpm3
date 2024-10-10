import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Đọc dữ liệu từ file CSV
file_path = '/ML_BTL/dataset4.csv' 
df = pd.read_csv(file_path)

#Lọc ra các cột cần thiết
selected_columns = ['PRICE', 'BEDROOMS', 'GARAGE', 'LAND_AREA', 'FLOOR_AREA', 'BUILD_YEAR', 'CBD_DIST']
df_filtered = df[selected_columns]

#Kiểm tra và xử lý giá trị thiếu
print("Giá trị NaN trong mỗi cột:\n", df_filtered.isnull().sum())

# Loại bỏ các hàng có giá trị NaN (hoặc bạn có thể sử dụng phương pháp điền giá trị)
df_filtered = df_filtered.dropna()

#Chia dữ liệu thành biến đầu vào (X) và biến mục tiêu (y)
X = df_filtered.drop('PRICE', axis=1)  # Các cột đầu vào
y = df_filtered['PRICE']  # Cột mục tiêu

#Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

#Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# In kết quả
print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')
