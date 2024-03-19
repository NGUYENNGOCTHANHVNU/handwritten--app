# Import các thư viện cần thiết
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

# Load dữ liệu MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Chuẩn hóa và chuyển đổi dữ liệu
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape dữ liệu để phù hợp với mạng neural network
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encode nhãn
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Xây dựng mô hình neural network
model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Đánh giá hiệu suất của mô hình trên tập test
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')

# Dự đoán một số ví dụ từ tập test
predictions = model.predict(X_test[:10])
predicted_labels = np.argmax(predictions, axis=1)

# Hiển thị các ví dụ và dự đoán tương ứng
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {predicted_labels[i]}')
    plt.axis('off')
plt.show()