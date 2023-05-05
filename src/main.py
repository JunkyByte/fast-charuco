import numpy as np
import torch
import cv2
import os
import cv2.aruco as aruco
import time
from tracking_utils import dcMultiTracker
from gridwindow import MagicGrid
from collections import deque
from typing import List, Optional
from multiprocessing import Queue
from queue import Full
opt_list_str = Optional[List[str]]


def main():
    from tracking_utils import FakeVideoCapture
    N = 1

    # caps = [FakeVideoCapture('deepcharuco/src/reference/board_image_240x240.jpg', (240, 240))
    #         for _ in range(N)]
    # mtxs = [np.eye(3) for _ in range(N)]
    # dists = [np.zeros((1, 5)) for _ in range(N)]

    caps = [FakeVideoCapture('deepcharuco/src/reference/samples_test/IMG_7412.png', (320, 240))
            for _ in range(N)]
    calib_data = np.load('../camera_params.npz')
    mtxs = [calib_data['camera_matrix'] for _ in range(N)]
    dists = [calib_data['distortion_coeffs'] for _ in range(N)]

    # Configuration
    board_name = 'DICT_4X4_50'
    row_count = 5
    col_count = 5
    square_len = 0.01
    marker_len = 0.0075

    deepc_path = './deepc.engine'
    refinenet_path = './refinenet.engine'

    n_ids = 16  # The number of corners (models pretrained use 16 for default board)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    multi_tracker = dcMultiTracker(deepc_path, refinenet_path, N, col_count,
                                   row_count, square_len, marker_len, n_ids)

    # Miscellaneous
    last_t = time.time()
    fps = deque([], maxlen=30)
    idx = 0
    draw = False

    if "DISPLAY" in os.environ and draw:
        w = MagicGrid(800, 800)
    while True:
        frames = [cap.read() for cap in caps]

        last_t = time.time()
        board_estim, frames = multi_tracker.track(frames, mtxs, dists, draw=draw)

        fps.append(1 / (time.time() - last_t))
        if idx > 0 and idx % 30 == 0:
            print('FPS TRACKER', np.mean(fps))

        # display the resulting frame
        if "DISPLAY" in os.environ and draw:
            if w.update(frames) & 0xFF == ord('q'):
                break
        idx += 1

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
