import numpy as np
class SoftmaxRegression:
    def __init__(self, so_vong_lap=1000, toc_do_hoc=0.1, so_class=10):
        self.so_vong_lap = so_vong_lap
        self.toc_do_hoc = toc_do_hoc
        self.so_class = so_class
        self.W = None
        self.b = None
        self.lich_su_loss = []
        self.lich_su_acc = []
        
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def tinh_loss(self, y_onehot, y_du_doan):
        eps = 1e-10
        y_du_doan = np.clip(y_du_doan, eps, 1 - eps)
        return -np.mean(np.sum(y_onehot * np.log(y_du_doan), axis=1))
    
    def fit(self, X, y):
        so_mau, so_dac_trung = X.shape
        if y.ndim == 1:
            y_onehot = np.eye(self.so_class)[y]
        else:
            y_onehot = y
        self.W = np.zeros((so_dac_trung, self.so_class))
        self.b = np.zeros(self.so_class)
        
        for epoch in range(self.so_vong_lap):
            z = np.dot(X, self.W) + self.b
            y_du_doan = self.softmax(z)
            
            dW = np.dot(X.T, (y_du_doan - y_onehot)) / so_mau
            db = np.sum(y_du_doan - y_onehot, axis=0) / so_mau
            
            self.W -= self.toc_do_hoc * dW
            self.b -= self.toc_do_hoc * db
            loss = self.tinh_loss(y_onehot, y_du_doan)
            nhan_du_doan = np.argmax(y_du_doan, axis=1)
            nhan_thuc_te = np.argmax(y_onehot, axis=1)
            acc = np.mean(nhan_du_doan == nhan_thuc_te)
            
            self.lich_su_loss.append(loss)
            self.lich_su_acc.append(acc)
            
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Vòng lặp {epoch+1}/{self.so_vong_lap} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")
                
    def predict(self, X):
        z = np.dot(X, self.W) + self.b
        y_du_doan = self.softmax(z)
        return np.argmax(y_du_doan, axis=1)

def bao_cao_danh_gia_nhieu_lop(y_true, y_pred, so_class=10):
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)  
    ds_precision, ds_recall, ds_f1 = [], [], []
    for c in range(so_class):
        TP = np.sum((y_true == c) & (y_pred == c))
        FP = np.sum((y_true != c) & (y_pred == c))
        FN = np.sum((y_true == c) & (y_pred != c))
        
        p = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        r = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        ds_precision.append(p)
        ds_recall.append(r)
        ds_f1.append(f1)
        
    accuracy = np.mean(y_true == y_pred)
    macro_precision = np.mean(ds_precision)
    macro_recall = np.mean(ds_recall)
    macro_f1 = np.mean(ds_f1)
    
    return accuracy, macro_precision, macro_recall, macro_f1