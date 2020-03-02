import numpy as np
import random
from sklearn.neighbors import NearestNeighbors

class KalmanFilter_attack:

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

    def create_measurement(self):
        np.random.seed(15)
        q = np.random.multivariate_normal([0, 0, 0, 0], self.Q)
        q = q.reshape(-1, 1)
        r = np.random.multivariate_normal([0, 0], self.R)
        r = r.reshape(-1, 1)
        self.z = self.H @ (self.mu_tplus1 + q) + r

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

    def NearestNeighbourAssociator(self, measurements):
        gate = np.sqrt(self.sigma_tplus1[0,0] + self.sigma_tplus1[1,1]) *3
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(measurements)
        distance, index = nbrs.kneighbors(self.predict_z.reshape(1, -1))
        if distance > gate:
            print("No data association...")
            self.z = []
            measurementID = None
        else:
            self.z = measurements[index].reshape(2, 1)
            measurementID = index[0][0]
        return measurementID

    # exhaustive search
    def Neighbour_measurement(self, measurements):
        gate = np.sqrt(self.sigma_tplus1[0, 0] + self.sigma_tplus1[1, 1]) * 3
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(measurements)
        distance, index = nbrs.kneighbors(self.predict_z.reshape(1, -1))
        num_set = sum(i <= gate for i in distance[0])
        if num_set == 0:
            print("No data association...")
            measurementID_set = np.array([])
        else:
            measurementID_set = index[0][:num_set]
        return measurementID_set

    # def NeighbourAssociator(self, measurements, old_kfz):
    #     gate = np.sqrt(self.sigma_tplus1[0,0] + self.sigma_tplus1[1,1]) *3
    #     nbrs = NearestNeighbors(n_neighbors= 5, algorithm='kd_tree').fit(measurements)
    #     distance, index = nbrs.kneighbors(self.predict_z.reshape(1, -1))
    #     num_set = sum(i <= gate for i in distance[0])
    #     if num_set == 0:
    #         print("No data association...")
    #         self.z = []
    #         measurementID = None
    #     else:
    #         measurements1 = measurements[index[0][:num_set]]
    #         num1 = len(measurements1)
    #         index1 = index[0][:num_set]
    #         nbrs1 = NearestNeighbors(n_neighbors = num1, algorithm='kd_tree').fit(measurements1)
    #         distance2, index2 = nbrs1.kneighbors(old_kfz.reshape(1, -1))
    #         self.z = measurements[index1[index2[0][num1-1]]].reshape(2, 1)
    #         measurementID = index1[index2[0][num1-1]]
    #     return measurementID

    # greedy search
    def NeighbourAssociation(self, measurements, old_kfz):
        gate = np.sqrt(self.sigma_tplus1[0, 0] + self.sigma_tplus1[1, 1]) * 3
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(measurements)
        distance, index = nbrs.kneighbors(self.predict_z.reshape(1, -1))
        num_set = sum(i <= gate for i in distance[0])
        if num_set == 0:
            print("No data association...")
        else:
            measurements1 = measurements[index[0][:num_set]]
            num1 = len(measurements1)
            index1 = index[0][:num_set]
            nbrs1 = NearestNeighbors(n_neighbors=num1, algorithm='kd_tree').fit(measurements1)
            distance2, index2 = nbrs1.kneighbors(old_kfz.reshape(1, -1))
            measurementID_set = index1[index2[0][:num1]]
        return measurementID_set

    # opposite direction search
    def NeighbourAssociation_direc(self, measurements, old_kfz, old_kfmu):
        gate = np.sqrt(self.sigma_tplus1[0, 0] + self.sigma_tplus1[1, 1]) * 3
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(measurements)
        distance, index = nbrs.kneighbors(self.predict_z.reshape(1, -1))
        num_set = sum(i <= gate for i in distance[0])
        if num_set == 0:
            print("No data association...")
        else:
            if (np.sign(old_kfmu[2])==np.sign(self.mu_t[2])) and (np.sign(old_kfmu[3])==np.sign(self.mu_t[3])):
                measurements1 = measurements[index[0][:num_set]]
                index1 = index[0][:num_set]
                mask = np.array([])
                direction_vector = measurements1 - self.predict_z.reshape(1, -1)
                for i, observation in enumerate(direction_vector):
                    if (np.sign(observation[0]) != np.sign(self.mu_t[2])) or (np.sign(observation[1]) != np.sign(self.mu_t[3])):
                        # measurements1 = measurements1[:i+1]
                        # measurementID_set = index1[:i+1]
                        # break
                        mask = np.append(mask,i)
                mask2 = mask.astype('int')
                measurements1 = measurements1[mask2]
                num1 = len(measurements1)
                index1 = index1[mask2]
                nbrs1 = NearestNeighbors(n_neighbors=num1, algorithm='kd_tree').fit(measurements1)
                distance2, index2 = nbrs1.kneighbors(old_kfz.reshape(1, -1))
                measurementID_set = index1[index2[0][:num1]]
            else:
                measurements1 = measurements[index[0][:num_set]]
                num1 = len(measurements1)
                index1 = index[0][:num_set]
                nbrs1 = NearestNeighbors(n_neighbors=num1, algorithm='kd_tree').fit(measurements1)
                distance2, index2 = nbrs1.kneighbors(old_kfz.reshape(1, -1))
                measurementID_set = index1[index2[0][:num1]]
        return measurementID_set


    def TimePropagate(self, transformation_matrix):
        x = self.mu_t[0, 0]
        y = self.mu_t[1, 0]
        newxyz = transformation_matrix @ np.array([[x], [y], [1]])
        newx = newxyz[0] / newxyz[2]
        newy = newxyz[1] / newxyz[2]
        self.mu_t[0] = newx
        self.mu_t[1] = newy

