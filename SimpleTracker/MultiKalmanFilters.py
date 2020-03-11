import numpy as np
from sklearn.neighbors import NearestNeighbors
from SimpleTracker.KalmanFilter import KalmanFilter


class MultiKalmanFilters:

    def __init__(self):
        self.KalmanFilters = []
        self.Measurements = []
        self.Unassociated = []
        self.Unassociated_prev = []

    def MaximumLikelihoodAssociator(self, primaryTrack, Measurements, gate_ll=1e-4):
        num_meas = Measurements.shape[0]
        KFs = self.KalmanFilters
        KFs.append(primaryTrack)
        KF_association = -1*np.ones(len(KFs))
        associated_meas = np.zeros(num_meas)
        measurementIDs = []
        LLs_all_kf = []
        for i, kf in enumerate(KFs):
            LLs = kf.LLs_with_all_measurements(Measurements)
            LLs_all_kf.append(LLs)
        LLs_all_kf = np.asarray(LLs_all_kf)
        # traverse the matrix
        sort_ind_raw = np.unravel_index(np.argsort(LLs_all_kf.ravel()), LLs_all_kf.shape)
        sort_ind_row = sort_ind_raw[0][::-1]    # from largest to smallest
        sort_ind_col = sort_ind_raw[1][::-1]    # from largest to smallest
        for i, KF in enumerate(KFs):
            KFs[i] = KF.clearAssociation()
        num_LLs = sort_ind_row.shape[0]
        for i in range(num_LLs):
            ir = sort_ind_row[i]
            ic = sort_ind_col[i]
            if LLs_all_kf[ir, ic] > gate_ll:
                if (KF_association[ir] == -1) and (associated_meas[ic] == 0):
                    KF_association[ir] = ic
                    associated_meas[ic] = 1
                    KFs[ir].DirectAssociator(Measurements[ic])
                else:
                    associated_meas[ic] = 1
            else:
                break
        self.KalmanFilters = KFs[:-1]
        primaryTrack = KFs[-1]
        primary_masurement_ID = KF_association[-1]
        if primary_masurement_ID == -1:
            primary_masurement_ID = None
        self.Unassociated = Measurements[np.where(associated_meas==0)]
        return primaryTrack, primary_masurement_ID

    def predict(self):
        for i, kf in enumerate(self.KalmanFilters):
            self.KalmanFilters[i].predict()
        return self

    def update(self):
        for i, kf in enumerate(self.KalmanFilters):
            self.KalmanFilters[i].update()
        return self

    def InitMultiKalmanFilters(self, primaryTrack):
        p_xy = primaryTrack.mu_t[0:2, 0]
        Unassociated_prev = self.Unassociated_prev
        Unassociated = self.Unassociated
        new_KFs = []
        for prev_ele in Unassociated_prev:
            for ele in Unassociated:
                Vx = ele[0] - prev_ele[0]
                Vy = ele[1] - prev_ele[1]
                V = np.sqrt((prev_ele-ele) @ (prev_ele-ele).T)
                D = np.sqrt((p_xy.T-ele) @ (p_xy.T-ele).T)
                if (V <= 30) and (D <= 300):
                    new_KFs.append(KalmanFilter(np.array([ele[0], ele[1], Vx, Vy]).reshape((4, 1)), np.diag([15**2, 15**2, 10**2, 10**2]), 5, 6))
        self.KalmanFilters = self.KalmanFilters + new_KFs
        return self

    def TimePropagate(self, transformation_matrix):
        Unassociated = self.Unassociated
        Unassociated_prev = np.zeros_like(Unassociated)
        for i, ele in enumerate(Unassociated):
            # propagate unassociated
            x = ele[0]
            y = ele[1]
            newxyz = transformation_matrix @ np.array([[x], [y], [1]])
            newx = newxyz[0] / newxyz[2]
            newy = newxyz[1] / newxyz[2]
            Unassociated_prev[i, 0] = newx
            Unassociated_prev[i, 1] = newy

        KFs = self.KalmanFilters
        for i, ele in enumerate(KFs):
            x = ele.mu_t[0, 0]
            y = ele.mu_t[1, 0]
            newxyz = transformation_matrix @ np.array([[x], [y], [1]])
            newx = newxyz[0] / newxyz[2]
            newy = newxyz[1] / newxyz[2]
            KFs[i].mu_t[0, 0] = newx
            KFs[i].mu_t[1, 0] = newy

        self.Unassociated_prev = Unassociated_prev
        self.KalmanFilters = KFs
        return self
