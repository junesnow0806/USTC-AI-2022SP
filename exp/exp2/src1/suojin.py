def predict_single(self, x):
    predict_y = 0.0
    for j in range(len(self.SV)):
        alpha_j = self.SV_alpha[j]
        yj = self.SV_label[j]
        xj = self.SV[j]
        predict_y += alpha_j * yj * self.KERNEL(xj, x)
    predict_y += self.b
    if predict_y >= 0:
        return 1
    else:
        return -1