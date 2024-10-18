from flask import Flask, render_template, request
import numpy as np
import pickle

# Tạo ứng dụng Flask
app = Flask(__name__)

# Load mô hình đã lưu (giả sử mô hình đã được lưu bằng pickle)
model = pickle.load(open('best_model.pkl', 'rb'))

# Trang chủ với form nhập liệu
@app.route('/')
def home():
    return render_template('index.html')

# Xử lý dữ liệu mới và dự đoán kết quả
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Lấy dữ liệu từ form (các trường tương ứng với đặc trưng của mô hình)
        bedrooms = float(request.form['bedrooms'])
        garage = float(request.form['garage'])
        land_area = float(request.form['land_area'])
        floor_area = float(request.form['floor_area'])
        build_year = float(request.form['build_year'])
        cbd_dist = float(request.form['cbd_dist'])
        
        # Đưa dữ liệu vào mô hình
        input_data = np.array([[bedrooms, garage, land_area, floor_area, build_year, cbd_dist]])
        prediction = model.predict(input_data)[0]
        prediction *= 16

        
        # Hiển thị kết quả dự đoán trên trang web
        return render_template('result.html', prediction_text=f'Giá nhà dự đoán: {prediction:.2f} VND')
        
# Chạy ứng dụng Flask
if __name__ == "__main__":
    app.run(debug=True)
  