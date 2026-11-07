import numpy as np
class LogisticRegression:
    def __init__(self, so_vong_lap=1000, toc_do_hoc=0.1):
        self.so_vong_lap = so_vong_lap
        self.toc_do_hoc = toc_do_hoc
        self.W = None
        self.b = None
        self.lich_su_loss = []
        self.lich_su_acc = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def tinh_loss(self, y_thuc_te, y_du_doan):
        eps = 1e-10 
        y_du_doan = np.clip(y_du_doan, eps, 1 - eps)
        return -np.mean(y_thuc_te * np.log(y_du_doan) + (1 - y_thuc_te) * np.log(1 - y_du_doan))
    
    def fit(self, X, y):
        so_mau, so_dac_trung = X.shape
        self.W = np.zeros(so_dac_trung)
        self.b = 0.0
        
        for epoch in range(self.so_vong_lap):
            z = np.dot(X, self.W) + self.b
            y_du_doan = self.sigmoid(z)
            dW = np.dot(X.T, (y_du_doan - y)) / so_mau
            db = np.sum(y_du_doan - y) / so_mau
            self.W -= self.toc_do_hoc * dW
            self.b -= self.toc_do_hoc * db
            loss = self.tinh_loss(y, y_du_doan)
            nhan_du_doan = (y_du_doan >= 0.5).astype(int)
            acc = np.mean(nhan_du_doan == y)
            self.lich_su_loss.append(loss)
            self.lich_su_acc.append(acc)
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Vòng lặp {epoch+1}/{self.so_vong_lap} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")
            
    def predict(self, X):
        z = np.dot(X, self.W) + self.b
        y_du_doan = self.sigmoid(z)
        return (y_du_doan >= 0.5).astype(int)

def bao_cao_danh_gia(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TN = np.sum((y_true == 0) & (y_pred == 0))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (TP + TN) / len(y_true)
    
    return accuracy, precision, recall, f1_score