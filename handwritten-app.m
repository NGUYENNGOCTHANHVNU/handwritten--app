# Import các thư viện cần thiết
import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load dữ liệu chữ số viết tay từ MNIST dataset
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Định nghĩa hàm dự đoán chữ số từ ảnh đầu vào
def predict_digit(image):
    prediction = model.predict(image)
    return prediction[0]

# Giao diện ứng dụng Streamlit
st.title('Ứng dụng Nhận dạng Chữ số viết tay')
st.write('Ứng dụng này sử dụng mô hình Random Forest để nhận dạng chữ số từ dữ liệu MNIST.')

# Đường dẫn cho người dùng tải lên ảnh chữ số
uploaded_file = st.file_uploader("Chọn một ảnh chứa chữ số viết tay...", type="png")

if uploaded_file is not None:
    # Đọc ảnh từ file người dùng tải lên
    image = np.array(uploaded_file)

    # Hiển thị ảnh
    st.image(image, caption='Ảnh vừa tải lên.', use_column_width=True)

    # Tiền xử lý ảnh (chuyển về định dạng phù hợp cho mô hình)
    # Ở đây bạn có thể thực hiện các bước tiền xử lý phù hợp với mô hình của bạn.

    # Dự đoán chữ số từ ảnh
    prediction = predict_digit(image)

    # Hiển thị kết quả dự đoán
    st.write(f"Dự đoán: {prediction}")
