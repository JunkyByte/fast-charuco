import numpy as np
import os
import torch
from functools import partial
import cv2
import cv2.aruco as aruco
import multiprocessing
import sys


sys.path.append('deepcharuco/src/')
sys.path.append('deepcharuco/src/models/')
from inference import pred_to_keypoints, extract_patches
from aruco_utils import draw_inner_corners
from models.model_utils import speedy_bargmax2d


def get_aruco_dict(board_name):
    return cv2.aruco.Dictionary_get(getattr(cv2.aruco, board_name))


def initialize_pool(board_name, col_count, row_count, square_len, marker_len):
    global aruco_dict, board, parameters
    parameters = aruco.DetectorParameters_create()
    aruco_dict = get_aruco_dict(board_name)
    board = cv2.aruco.CharucoBoard_create(squaresX=col_count,
                                          squaresY=row_count,
                                          squareLength=square_len,
                                          markerLength=marker_len,
                                          dictionary=aruco_dict)


def estimate_board(gray, mtx, dist):
    corners, ids, rej = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    corners, ids, _, _ = aruco.refineDetectedMarkers(gray, board, corners,
                                                     ids, rej, mtx, dist)
    ret = False
    rvec = None
    tvec = None
    if np.any(ids is not None):
        ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board,
                                                  mtx, dist, rvec, tvec)
    return ret, rvec, tvec, corners


class MultiTracker:
    def __init__(self, board_name, col_count, row_count,
                 square_len, marker_len, nproc=os.cpu_count()):
        self.col_count = col_count
        self.row_count = row_count
        self.square_len = square_len
        self.marker_len = marker_len
        ctx = multiprocessing.get_context('spawn')
        self.pool = ctx.Pool(processes=nproc,
                             initializer=partial(initialize_pool, board_name,
                                                 col_count, row_count,
                                                 square_len, marker_len))

    def track(self, frames, mtxs, dists, draw=False):
        grays = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

        # Populate tvecs and rvecs for each camera that found the board
        args = ((gray, mtxs[i], dists[i]) for i, gray in enumerate(grays))
        ret, rvec, tvec, corners = zip(*self.pool.starmap(estimate_board, args))
        board_estim = list(zip(ret, rvec, tvec))

        if draw:
            # Draw corners on images
            for i, corners_i in enumerate(corners):
                aruco.drawDetectedMarkers(frames[i], corners_i)

            for i, (ret, rvec, tvec) in enumerate(board_estim):
                if not ret:
                    continue

                cv2.drawFrameAxes(frames[i], mtxs[i], dists[i], rvec, tvec, 0.01)
        return board_estim, frames


import onnxruntime
class dcMultiTracker:
    def __init__(self, deepc_path, refinenet_path, col_count, row_count,
                 square_len, marker_len, n_ids=16, device='cuda'):
        self.deepc_sess = onnxruntime.InferenceSession('./deepc.onnx',
                                                       providers=['CUDAExecutionProvider',
                                                                  'CPUExecutionProvider'])
        self.deepc_name = self.deepc_sess.get_inputs()[0].name

        self.refinenet_sess = onnxruntime.InferenceSession('./refinenet.onnx',
                                                       providers=['CUDAExecutionProvider',
                                                                  'CPUExecutionProvider'])
        self.refinenet_name = self.refinenet_sess.get_inputs()[0].name

        self.n_ids = n_ids
        self.col_count = col_count
        self.row_count = row_count
        self.square_len = square_len
        self.marker_len = marker_len

        # Create inner corners board points
        inn_rc = np.arange(1, self.row_count)
        inn_cc = np.arange(1, self.col_count)
        self.object_points = np.zeros(((self.col_count - 1) * (self.row_count - 1), 3), np.float32)
        self.object_points[:, :2] = np.array(np.meshgrid(inn_rc, inn_cc)).reshape((2, -1)).T * self.square_len

    def solve_pnp(self, keypoints, camera_matrix, dist_coeffs):
        if keypoints.shape[0] < 4:
            return False, None, None

        image_points = keypoints[:, :2].astype(np.float32)
        object_points_found = self.object_points[keypoints[:, 2].astype(int)]

        ret, rvec, tvec = cv2.solvePnP(object_points_found, image_points,
                                       camera_matrix, dist_coeffs)
        return ret, rvec, tvec

    def track(self, frames, mtxs, dists, draw=False):
        import time
        imgs_gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                               for frame in frames])
        imgs_gray = (imgs_gray.astype(np.float32) - 128) / 255  # Well we started with this one so...

        # Run batched inference with deepc
        with torch.no_grad():
            inf_imgs = np.expand_dims(imgs_gray, axis=1)
            bloc_hat, bids_hat = self.deepc_sess.run(None, {self.deepc_name: inf_imgs})

        bkpts_hat = []
        bids_found = []
        patches = []
        for i, (loc_hat, ids_hat) in enumerate(zip(bloc_hat, bids_hat)):
            loc_hat = loc_hat[None, ...]
            ids_hat = ids_hat[None, ...]
            k, id = pred_to_keypoints(loc_hat, ids_hat, self.n_ids)
            bkpts_hat.append(k)
            bids_found.append(id)

            if draw:
                frames[i] = draw_inner_corners(frames[i], k, id, radius=3,
                                               draw_ids=True, color=(0, 0, 255))

            patches.append(extract_patches(imgs_gray[i][None, ...], k))

        # Run batched inference with refinenet
        with torch.no_grad():
            patches_dims = [p.shape[0] for p in patches]
            patches = np.vstack(patches)

            if patches.ndim == 3:
                patches = np.expand_dims(patches, axis=1)
            bloc_hat = self.refinenet_sess.run(None, {self.refinenet_name: patches})[0][:, 0]

        res = []
        c = 0
        t = time.time()
        for i, (kpts_hat, ids_found) in enumerate(zip(bkpts_hat, bids_found)):
            # Recover loc_hat using dims
            t = time.time()
            loc_hat = bloc_hat[c: c + patches_dims[i]]
            c += patches_dims[i]

            refined_kpts = (speedy_bargmax2d(loc_hat) - 32) / 8 + kpts_hat
            keypoints = np.array([[k[0], k[1], idx] for k, idx in
                                  zip(refined_kpts, ids_found)])
            ret, rvec, tvec = self.solve_pnp(keypoints, mtxs[i], dists[i])

            # Draw refinenet refined corners in yellow
            if draw:
                frames[i] = draw_inner_corners(frames[i], refined_kpts,
                                               ids_found, draw_ids=False,
                                               radius=1, color=(0, 255, 255))

            if draw and ret:  # Draw axis
                cv2.drawFrameAxes(frames[i], mtxs[i], dists[i], rvec, tvec, 0.01, 2)

            res.append((ret, rvec, tvec))
        return res, frames


class FakeVideoCapture:
    def __init__(self, img_path, res):
        self.img = cv2.resize(cv2.imread(img_path), res)

    def read(self):
        return self.img.copy()

    def release(self):
        pass
