import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2


# window-tuple: (min_r, max_r, min_c, max_c)
def init_Track_From_Groundtruth(trans_matrices, frame_number: int, window: tuple, groundtruth_path=None, Image=None):
    Candidate_tracks = []
    trans_matrix = trans_matrices[frame_number - 2][0]
    # There is a offset...
    frame_number = frame_number + 99
    if groundtruth_path is None:
        groundtruth = pd.read_csv("Models/ground-truth.csv")
    else:
        groundtruth = pd.read_csv(groundtruth_path)

    min_x = window[2]
    max_x = window[3]
    min_y = window[0]
    max_y = window[1]
    width = max_x - min_x
    height = max_y - min_y

    # gather coordinates in previous frame
    gt_by_time = groundtruth.groupby("FRAME_NUMBER")
    prev_df = gt_by_time.get_group(frame_number-1)
    prev_Xs = np.array(prev_df.X)
    prev_Ys = np.array(prev_df.Y)
    prev_ids = np.asarray(prev_df.id)
    # Need to do a projective transformation
    #print(trans_matrix)
    ttmp = np.array([prev_Xs, prev_Ys, np.ones(len(prev_Xs))])
    #print(ttmp.shape)
    new_xyz = trans_matrix @ np.array([prev_Xs, prev_Ys, np.ones(len(prev_Xs))])
    prev_Xs_tonow = new_xyz[0, :] / new_xyz[2, :]
    prev_Ys_tonow = new_xyz[1, :] / new_xyz[2, :]

    # gather coordinats in current frame
    now_df = gt_by_time.get_group(frame_number)
    now_Xs = np.array(now_df.X)
    now_Ys = np.array(now_df.Y)
    now_ids = np.asarray(now_df.id)
    num_of_detections = len(now_df)

    for i in range(num_of_detections):
        x = now_Xs[i] - min_x
        y = now_Ys[i] - min_y
        tmp_id = now_ids[i]
        if (x > 200) & (x < width-200) & (y > 200) & (y < height-200):
            ind = np.where(prev_ids == tmp_id)
            if len(ind) > 0:
                vel_x = x - (prev_Xs_tonow[ind][0] - min_x)
                vel_y = y - (prev_Ys_tonow[ind][0] - min_y)
                if (np.abs(vel_x) >= 5) | (np.abs(vel_y) >= 5):
                    Candidate_tracks.append([x, y, vel_x, vel_y])

    if Image is not None:
        Image_part = np.repeat(np.expand_dims(Image[min_y:max_y+1, min_x:max_x+1], -1), 3, axis=2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, track in enumerate(Candidate_tracks):
            cv2.putText(Image_part, str(i), (track[0], track[1]), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        plt.figure()
        plt.imshow(Image_part)
        plt.show()

    return Candidate_tracks