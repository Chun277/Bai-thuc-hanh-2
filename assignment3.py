import numpy as np
import idx2numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
x_train_raw = idx2numpy.convert_from_file("train-images.idx3-ubyte")
y_train = idx2numpy.convert_from_file("train-labels.idx1-ubyte")
x_test_raw = idx2numpy.convert_from_file("t10k-images.idx3-ubyte")
y_test = idx2numpy.convert_from_file("t10k-labels.idx1-ubyte")
x_train = x_train_raw.reshape(x_train_raw.shape[0], -1) / 255.0
x_test = x_test_raw.reshape(x_test_raw.shape[0], -1) / 255.0
mask_train = (y_train <= 1)
mask_test = (y_test <= 1)
mo_hinh_logistic = LogisticRegression(max_iter=1000)
mo_hinh_logistic.fit(x_train[mask_train], y_train[mask_train])
y_pred_log = mo_hinh_logistic.predict(x_test[mask_test])
print("Logistic regression")
print(classification_report(y_test[mask_test], y_pred_log, digits=4))
mo_hinh_softmax = LogisticRegression(max_iter=1000)
mo_hinh_softmax.fit(x_train, y_train) 
y_pred_soft = mo_hinh_softmax.predict(x_test)
print("Softmax regression")
print(classification_report(y_test, y_pred_soft, digits=4))