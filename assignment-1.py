import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression, bao_cao_danh_gia

x_train_raw = idx2numpy.convert_from_file("train-images.idx3-ubyte")
y_train_raw = idx2numpy.convert_from_file("train-labels.idx1-ubyte")
x_test_raw = idx2numpy.convert_from_file("t10k-images.idx3-ubyte")
y_test_raw = idx2numpy.convert_from_file("t10k-labels.idx1-ubyte")

def filter_data(images, labels, condition):
    new_images = images[labels == condition]
    new_labels = labels[labels == condition]
    return new_images, new_labels

x_train_0, y_train_0 = filter_data(x_train_raw, y_train_raw, 0)
x_train_1, y_train_1 = filter_data(x_train_raw, y_train_raw, 1)
x_train = np.concatenate((x_train_0, x_train_1), axis=0)
y_train = np.concatenate((y_train_0, y_train_1), axis=0)

x_test_0, y_test_0 = filter_data(x_test_raw, y_test_raw, 0)
x_test_1, y_test_1 = filter_data(x_test_raw, y_test_raw, 1)
x_test = np.concatenate((x_test_0, x_test_1), axis=0)
y_test = np.concatenate((y_test_0, y_test_1), axis=0)

N_train = x_train.shape[0]
x_train_flat = x_train.reshape(N_train, -1) / 255.0

N_test = x_test.shape[0]
x_test_flat = x_test.reshape(N_test, -1) / 255.0
mo_hinh = LogisticRegression(so_vong_lap=1000, toc_do_hoc=0.1)
mo_hinh.fit(x_train_flat, y_train)

y_pred = mo_hinh.predict(x_test_flat)

# Sử dụng hàm tự code bằng Numpy
acc, prec, rec, f1 = bao_cao_danh_gia(y_test, y_pred)
print("\nKết quả đánh giá")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(mo_hinh.so_vong_lap), mo_hinh.lich_su_loss, color='red')
plt.title("Biến thiên Loss")
plt.xlabel("Epochs")

plt.subplot(1, 2, 2)
plt.plot(range(mo_hinh.so_vong_lap), mo_hinh.lich_su_acc, color='blue')
plt.title("Biến thiên Accuracy")
plt.xlabel("Epochs")

plt.tight_layout()
plt.show()