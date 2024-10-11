import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Tải dữ liệu
file_path = r'C:\Users\Admin\Documents\GitHub\nhom16.ktpm3\dtset.csv'  # Thay đổi đường dẫn nếu cần

data = pd.read_csv(file_path)

# Kiểm tra và xử lý giá trị thiếu (nếu có)
if data.isnull().sum().any():
    data.fillna(data.mean(), inplace=True)

# 2. Tách các đặc trưng và biến mục tiêu
X = data.drop(columns=['PRICE'])
y = data['PRICE']

# 3. Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% huấn luyện, 20% kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Chuẩn hóa các đặc trưng sử dụng StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Định nghĩa các mô hình cơ bản cho stacking
base_models = [
    
    
]

# 6. Stacking Regressor sử dụng mô hình Ridge làm bộ ước lượng cuối cùng
stacking_model = StackingRegressor(estimators=base_models, final_estimator=Ridge(alpha=1.0), n_jobs=-1)
stacking_model.fit(X_train_scaled, y_train)

# 7. Dự đoán và đánh giá mô hình stacking
y_pred_stacking = stacking_model.predict(X_test_scaled)

mse_stacking = mean_squared_error(y_test, y_pred_stacking)
r2_stacking = r2_score(y_test, y_pred_stacking)

# 8. Hiển thị các chỉ số đánh giá cho mô hình stacking
print(f'MSE (Stacking): {mse_stacking:.2f}')
print(f'R² Score (Stacking): {r2_stacking:.4f}')