import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from softmax_regression import SoftmaxRegression, bao_cao_danh_gia_nhieu_lop

def convert_to_onehot_vector(labels):
    N = labels.shape[0]
    total_classes = labels.max() + 1
    oh_labels = np.zeros((N, total_classes))
    oh_labels[np.arange(N), labels] = 1
    return oh_labels

x_train_raw = idx2numpy.convert_from_file("train-images.idx3-ubyte")
y_train_raw = idx2numpy.convert_from_file("train-labels.idx1-ubyte")
x_test_raw = idx2numpy.convert_from_file("t10k-images.idx3-ubyte")
y_test_raw = idx2numpy.convert_from_file("t10k-labels.idx1-ubyte")

N_train = x_train_raw.shape[0]
x_train_flat = x_train_raw.reshape(N_train, -1) / 255.0
N_test = x_test_raw.shape[0]
x_test_flat = x_test_raw.reshape(N_test, -1) / 255.0
mo_hinh = SoftmaxRegression(so_vong_lap=1000, toc_do_hoc=0.5, so_class=10)

y_train_onehot = convert_to_onehot_vector(y_train_raw)
mo_hinh.fit(x_train_flat, y_train_onehot)

y_pred = mo_hinh.predict(x_test_flat)

acc, prec, rec, f1 = bao_cao_danh_gia_nhieu_lop(y_test_raw, y_pred, so_class=10)
print("\nKết quả đánh giá")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(mo_hinh.so_vong_lap), mo_hinh.lich_su_loss, color='green')
plt.title("Biến thiên Loss")
plt.xlabel("Epochs")

plt.subplot(1, 2, 2)
plt.plot(range(mo_hinh.so_vong_lap), mo_hinh.lich_su_acc, color='orange')
plt.title("Biến thiên Accuracy")
plt.xlabel("Epochs")

plt.tight_layout()
plt.show()