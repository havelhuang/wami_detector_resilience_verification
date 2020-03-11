import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import multivariate_normal


class KalmanFilter:

    def __init__(self, init_mu, init_sigma, Q_sigma, R_sigma):
        self.mu_t = init_mu
        self.mu_tplus1 = init_mu
        self.sigma_t = init_sigma
        self.sigma_tplus1 = init_sigma
        self.predict_z = []
        self.z = []
        self.uncertainty = []
        # position-x, position-y, velocity-x, velocity-y
        self.F = np.asarray([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = (Q_sigma ** 2) * np.array([[1/3, 0, 1/2, 0],
                                     [0, 1/3, 0, 1/2],
                                     [1/2, 0, 1, 0],
                                     [0, 1/2, 0, 1]])
        self.R = R_sigma ** 2 * np.eye(2)

    def predict(self):
        self.mu_tplus1 = self.F @ self.mu_t
        self.sigma_tplus1 = self.F @ self.sigma_t @ self.F.transpose() + self.Q
        self.predict_z = self.H @ self.mu_tplus1
        return self.mu_tplus1, self.sigma_tplus1

    def update(self):
        if len(self.z) > 0:
            y = self.z - self.predict_z
            S = self.H @ self.sigma_tplus1 @ self.H.transpose() + self.R
            K = self.sigma_tplus1 @ self.H.transpose() @ np.linalg.inv(S)
            mu_t_t = self.mu_tplus1 + K @ y
            sigma_t_t = (np.eye(4) - K @ self.H) @ self.sigma_tplus1
        else:
            mu_t_t = self.mu_tplus1
            sigma_t_t = self.sigma_tplus1
        uncert = np.sqrt(self.sigma_tplus1[0,0] + self.sigma_tplus1[1,1]) *3
        self.uncertainty.append(uncert)
        self.mu_t = mu_t_t
        self.sigma_t = sigma_t_t
        return mu_t_t, sigma_t_t

    def clearAssociation(self):
        self.z = []
        return self

    def DirectAssociator(self, measurement):
        self.z = measurement.reshape(2, 1)
        return self

    def NearestNeighbourAssociator(self, measurements, gate_ll=1e-4):
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(measurements)
        distance, index = nbrs.kneighbors(self.predict_z.reshape(1, -1))
        z_ = measurements[index].reshape(2, 1)
        S = self.H @ self.sigma_tplus1 @ self.H.transpose() + self.R
        ll = multivariate_normal.pdf(np.squeeze(z_), np.squeeze(self.predict_z), S)
        if ll <= gate_ll:
            print("No data association...")
            self.z = []
            measurementID = None
        else:
            self.z = z_
            measurementID = index[0][0]
        return measurementID

    def LLs_with_all_measurements(self, measurements):
        LLs = []
        S = self.H @ self.sigma_tplus1 @ self.H.transpose() + self.R
        for measurement in measurements:
            z_ = measurement.reshape(2, 1)
            ll = multivariate_normal.pdf(np.squeeze(z_), np.squeeze(self.predict_z), S)
            LLs.append(ll)
        return np.array(LLs)


    def TimePropagate(self, transformation_matrix):
        x = self.mu_t[0, 0]
        y = self.mu_t[1, 0]
        newxyz = transformation_matrix @ np.array([[x], [y], [1]])
        newx = newxyz[0] / newxyz[2]
        newy = newxyz[1] / newxyz[2]
        self.mu_t[0] = newx
        self.mu_t[1] = newy

