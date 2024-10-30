from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Load dữ liệu và huấn luyện mô hình (mô phỏng)
file_path = r'C:\Users\Admin\Desktop\ML\last\housing.csv'
housing_data = pd.read_csv(file_path, delim_whitespace=True)
column_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
housing_data.columns = column_names

# Chuẩn bị dữ liệu và huấn luyện mô hình
X = housing_data.drop("MEDV", axis=1)
y = housing_data["MEDV"]
columns_to_drop = ['CHAS', 'ZN', 'RAD', 'INDUS', 'PTRATIO']
X_selected = X.drop(columns_to_drop, axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Khởi tạo và huấn luyện các mô hình
linear_reg = LinearRegression()
ridge_reg = Ridge(alpha=1.0)
mlp_reg = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
stacking_regressor = StackingRegressor(
    estimators=[('linear', linear_reg), ('ridge', ridge_reg), ('mlp', mlp_reg)],
    final_estimator=LinearRegression()
)

linear_reg.fit(X_scaled, y)
ridge_reg.fit(X_scaled, y)
mlp_reg.fit(X_scaled, y)
stacking_regressor.fit(X_scaled, y)

# Trang chủ với form nhập liệu
@app.route('/')
def home():
    return render_template('index.html')

# Xử lý dự đoán khi gửi form
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    features = [float(data[column]) for column in X_selected.columns]
    scaled_features = scaler.transform([features])
    
    # Dự đoán với từng mô hình và nhân với 1000 để chuyển từ nghìn USD sang USD
    prediction_linear = linear_reg.predict(scaled_features)[0] * 1000
    prediction_ridge = ridge_reg.predict(scaled_features)[0] * 1000
    prediction_mlp = mlp_reg.predict(scaled_features)[0] * 1000
    prediction_stacking = stacking_regressor.predict(scaled_features)[0] * 1000
    
    # Trả về kết quả của từng mô hình
    return jsonify({
        "predicted_price_linear": round(prediction_linear, 2),
        "predicted_price_ridge": round(prediction_ridge, 2),
        "predicted_price_mlp": round(prediction_mlp, 2),
        "predicted_price_stacking": round(prediction_stacking, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
