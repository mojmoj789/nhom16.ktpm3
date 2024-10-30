# Import các thư viện cần thiết
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dữ liệu
file_path = r'C:\Users\Admin\Desktop\ML\last\housing.csv'
housing_data = pd.read_csv(file_path, delim_whitespace=True)

# Gán tên cột
column_names = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD",
    "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]
housing_data.columns = column_names

# Phân tích dữ liệu bằng biểu đồ
plt.figure(figsize=(12, 8))
sns.heatmap(housing_data.corr(), annot=True, cmap='coolwarm')
plt.title('Ma trận tương quan giữa các biến')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(housing_data['MEDV'], kde=True, bins=30)
plt.title('Phân bố giá trị MEDV (giá nhà trung bình)')
plt.xlabel('MEDV')
plt.ylabel('Tần số')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=housing_data.sample(frac=0.5, random_state=42), x='RM', y='MEDV', alpha=0.8)
plt.title('Quan hệ giữa số phòng trung bình (RM) và giá nhà (MEDV)')
plt.xlabel('Số phòng trung bình (RM)')
plt.ylabel('Giá nhà trung bình (MEDV)')
plt.show()


# Chuẩn bị dữ liệu
X = housing_data.drop("MEDV", axis=1)  # Các biến độc lập
y = housing_data["MEDV"]  # Biến mục tiêu

# Loại bỏ các cột ít quan trọng: CHAS, ZN, RAD, INDUS, PTRATIO
columns_to_drop = ['CHAS', 'ZN', 'RAD', 'INDUS', 'PTRATIO']
X_selected = X.drop(columns_to_drop, axis=1)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train_selected, X_test_selected, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_selected_scaled = scaler.fit_transform(X_train_selected)
X_test_selected_scaled = scaler.transform(X_test_selected)

# Hồi quy tuyến tính
linear_reg = LinearRegression()
linear_reg.fit(X_train_selected_scaled, y_train)
y_pred_linear = linear_reg.predict(X_test_selected_scaled)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# Hồi quy Ridge
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train_selected_scaled, y_train)
y_pred_ridge = ridge_reg.predict(X_test_selected_scaled)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

# Mạng neural với biểu đồ hàm mất mát
mlp = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
mlp.fit(X_train_selected_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_selected_scaled)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)

# Vẽ biểu đồ hàm mất mát sau mỗi vòng lặp
plt.plot(mlp.loss_curve_)
plt.title('Hàm mất mát của MLPRegressor')
plt.xlabel('Số vòng lặp')
plt.ylabel('Giá trị hàm mất mát')
plt.show()

# Stacking Regressor
base_models = [
    ('linear', LinearRegression()),
    ('ridge', Ridge(alpha=1.0)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42))
]
meta_model = LinearRegression()
stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=meta_model)
stacking_regressor.fit(X_train_selected_scaled, y_train)
y_pred_stacking = stacking_regressor.predict(X_test_selected_scaled)
mse_stacking = mean_squared_error(y_test, y_pred_stacking)
r2_stacking = r2_score(y_test, y_pred_stacking)

# In kết quả
print("Linear Regression: MSE = {:.2f}, R^2 = {:.2f}".format(mse_linear, r2_linear))
print("Ridge Regression: MSE = {:.2f}, R^2 = {:.2f}".format(mse_ridge, r2_ridge))
print("Neural Network: MSE = {:.2f}, R^2 = {:.2f}".format(mse_mlp, r2_mlp))
print("Stacking Regressor: MSE = {:.2f}, R^2 = {:.2f}".format(mse_stacking, r2_stacking))

# Biểu đồ so sánh kết quả các mô hình
models = ['Linear Regression', 'Ridge Regression', 'Neural Network', 'Stacking Regressor']
mse_values = [mse_linear, mse_ridge, mse_mlp, mse_stacking]
r2_values = [r2_linear, r2_ridge, r2_mlp, r2_stacking]

plt.figure(figsize=(12, 6))
plt.bar(models, mse_values, color='skyblue')
plt.title('So sánh MSE của các mô hình')
plt.xlabel('Mô hình')
plt.ylabel('MSE')
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(models, r2_values, color='lightgreen')
plt.title('So sánh R^2 của các mô hình')
plt.xlabel('Mô hình')
plt.ylabel('R^2')
plt.show()