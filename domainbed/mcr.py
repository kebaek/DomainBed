import numpy as np
import torch
from itertools import combinations

class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, eps=0.01):
        super(MaximalCodingRateReduction, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps

    def compute_discrimn_loss_empirical(self, W):
        """Empirical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_empirical(self, W, Pi):
        """Empirical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).cuda()
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += log_det * trPi / float(m)
        return compress_loss / 2.

    def compute_discrimn_loss_theoretical(self, W):
        """Theoretical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_theoretical(self, W, Pi):
        """Theoretical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).cuda()
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += trPi / (2. * m) * log_det
        return compress_loss

    def forward(self, X, Y, num_classes=None):
        if num_classes is None:
            num_classes = (Y.max() + 1).cpu()
        W = X.T
        Pi = label_to_membership(Y.cpu().numpy(), num_classes)
        Pi = torch.tensor(Pi, dtype=torch.float32).cuda()

        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
        compress_loss_empi = self.compute_compress_loss_empirical(W, Pi)

        total_loss_empi = (self.gam2 * -discrimn_loss_empi + compress_loss_empi)
        return (total_loss_empi,0,0)

class MutualInformation(torch.nn.Module):
    def __init__(self, eps=0.01):
        super(MutualInformation, self).__init__()
        self.eps = eps

    def forward(self, D1, D2):
        z = torch.cat((D1,D2), 0)
        m,p = z.shape
        m1, _ = D1.shape
        m2, _ = D2.shape
        I = torch.eye(p).cpu()
        scalar = p / (m * self.eps)
        scalar1 = p / (m1 * self.eps)
        scalar2 = p / (m2 * self.eps)
        ld = torch.logdet(I + scalar * (z.T).matmul(z)) / 2.
        ld1 = m1 * torch.logdet(I + scalar1 * (D1.T).matmul(D1)) / (2. * m)
        ld2 = m2 * torch.logdet(I + scalar2 * (D2.T).matmul(D2)) / (2. * m)
        return (ld - ld1 - ld2).cuda()


def label_to_membership(targets, num_classes=None):
    """Generate a true membership matrix, and assign value to current Pi.

    Parameters:
        targets (np.ndarray): matrix with one hot labels

    Return:
        Pi: membership matirx, shape (num_classes, num_samples, num_samples)

    """
    targets = one_hot(targets, num_classes)
    num_samples, num_classes = targets.shape
    Pi = np.zeros(shape=(num_classes, num_samples, num_samples))
    for j in range(len(targets)):
        k = np.argmax(targets[j])
        Pi[k, j, j] = 1.
    return Pi

def one_hot(x, K):
    """Turn labels x into one hot vector of K classes. """
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)
