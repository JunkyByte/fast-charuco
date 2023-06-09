import numpy as np
import cv2
import sys
import onnxruntime

sys.path.append('deepcharuco/src/')
sys.path.append('deepcharuco/src/models/')
from inference import pred_to_keypoints, extract_patches
from aruco_utils import draw_inner_corners
from models.model_utils import speedy_bargmax2d


class dcMultiTracker:
    def __init__(self, deepc_path, refinenet_path, col_count, row_count,
                 square_len, marker_len, n_ids=16, device='cuda'):
        self.deepc_sess = onnxruntime.InferenceSession('./deepc.onnx',
                                                       providers=['CUDAExecutionProvider',
                                                                  'CPUExecutionProvider'])
        self.deepc_name = self.deepc_sess.get_inputs()[0].name

        self.ref_sess = onnxruntime.InferenceSession('./refinenet.onnx',
                                                     providers=['CUDAExecutionProvider',
                                                                'CPUExecutionProvider'])
        self.refinenet_name = self.ref_sess.get_inputs()[0].name

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
        imgs_gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                              for frame in frames])
        imgs_gray = (imgs_gray.astype(np.float32) - 128) / 255  # Well we started with this one so...

        # Run batched inference with deepc
        inf_imgs = np.expand_dims(imgs_gray, axis=1)
        bloc_hat, bids_hat = self.deepc_sess.run(None, {self.deepc_name: inf_imgs})

        bkpts_hat = []
        bids_found = []
        patches = []
        for i, (loc_hat, ids_hat) in enumerate(zip(bloc_hat, bids_hat)):
            k, id = pred_to_keypoints(loc_hat[None, ...], ids_hat[None, ...], self.n_ids)
            bkpts_hat.append(k)
            bids_found.append(id)

            if draw:
                frames[i] = draw_inner_corners(frames[i], k, id, radius=3,
                                               draw_ids=True, color=(0, 0, 255))

            patches.append(extract_patches(imgs_gray[i][None, ...], k))

        # Run batched inference with refinenet
        patches_dims = [p.shape[0] for p in patches]
        patches = np.vstack(patches)

        if patches.ndim == 3:
            patches = np.expand_dims(patches, axis=1)
        bloc_hat = self.ref_sess.run(None, {self.refinenet_name: patches})[0][:, 0]

        res = []
        c = 0
        for i, (kpts_hat, ids_found) in enumerate(zip(bkpts_hat, bids_found)):
            # Recover loc_hat using dims
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
