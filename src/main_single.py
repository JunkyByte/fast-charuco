import numpy as np
import torch
import cv2
import os
import time
from gridwindow import MagicGrid
from collections import deque
import sys

sys.path.append('deepcharuco/src/')
sys.path.append('deepcharuco/src/models/')
from inference import load_models, infer_image, solve_pnp


def inf_single_dc(img, n_ids, deepc, refinenet, col_count, row_count,
                  square_len, camera_matrix, dist_coeffs, draw=False):
    keypoints, out_img = infer_image(img, n_ids, deepc, refinenet, draw_pred=draw)
    ret, rvec, tvec = solve_pnp(keypoints, col_count, row_count, square_len,
                                camera_matrix, dist_coeffs)
    return (ret, rvec, tvec), out_img


def main():
    from tracking_utils import FakeVideoCapture
    N = 4

    caps = [FakeVideoCapture('deepcharuco/src/reference/samples_test/IMG_7412.png', (320, 240))
            for _ in range(N)]
    calib_data = np.load('../camera_params.npz')
    mtxs = [calib_data['camera_matrix'] for _ in range(N)]
    dists = [calib_data['distortion_coeffs'] for _ in range(N)]

    # Configuration
    row_count = 5
    col_count = 5
    square_len = 0.01

    deepc_path = 'deepcharuco/src/reference/longrun-epoch=99-step=369700.ckpt'
    refinenet_path = 'deepcharuco/src/reference/second-refinenet-epoch-100-step=373k.ckpt'
    n_ids = 16  # The number of corners (models pretrained use 16 for default board)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    deepc, refinenet = load_models(deepc_path, refinenet_path, n_ids, device=device)
    img = cv2.imread('deepcharuco/src/reference/samples_test/IMG_7412.png')

    # Miscellaneous
    last_t = time.time()
    fps = deque([], maxlen=30)
    idx = 0

    if "DISPLAY" in os.environ:
        w = MagicGrid(800, 800)
    while True:
        frames = [cap.read() for cap in caps]

        last_t = time.time()
        for _ in range(N):
            board_estim, frames = inf_single_dc(img, n_ids, deepc, refinenet,
                                                col_count, row_count,
                                                square_len, mtxs[0], dists[0],
                                                draw=False)

        fps.append(1 / (time.time() - last_t))
        if idx > 0 and idx % 100 == 0:
            print('FPS TRACKER', np.mean(fps))

        # display the resulting frame
        if "DISPLAY" in os.environ:
            if w.update(frames) & 0xFF == ord('q'):
                break
        idx += 1

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
