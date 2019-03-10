from slam import SLAM
import cv2
import numpy as np
from display import Display2D, Display3D
import argparse

def create_camera_params(cap):
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    CNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    F = float(args.f)
    if W > 1024:
        downscale = 1024.0 / W
        F *= downscale
        H = int(H * downscale)
        W = 1024
    print("using camera %dx%d with F %f" % (W, H, F))

    # camera intrinsics
    K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])
    Kinv = np.linalg.inv(K)

    return W, H, K, CNT

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", nargs="?", type=str, help="path to the video")
    parser.add_argument("pose_path",  nargs="?", type=str, help="pose file path", default=None)
    parser.add_argument("--headless", type=bool, default=None)
    parser.add_argument("-f", type=int, default=525)
    parser.add_argument("--seek", action='store_true', default=False)
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video_path)
    W, H, K, CNT = create_camera_params(cap)
    if args.seek:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.seek))
    disp2d, disp3d = None, None
    if args.headless is None:
        disp2d = Display2D(W, H)
        disp3d = Display3D()
    slam = SLAM(W, H, K)

    """
    mapp.deserialize(open('map.json').read())
    while 1:
      disp3d.paint(mapp)
      time.sleep(1)
    """

    gt_pose = None
    if args.pose_path:
        gt_pose = np.load(args.pose_path)['pose']
        # add scale param?
        gt_pose[:, :3, 3] *= 50

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (W, H))

        print("\n*** frame %d/%d ***" % (i, CNT))
        if ret == True:
            slam.process_frame(frame, None if gt_pose is None else np.linalg.inv(gt_pose[i]))
        else:
            break

        # 3-D display
        if disp3d is not None:
            disp3d.paint(slam.mapp)

        if disp2d is not None:
            img = slam.mapp.frames[-1].annotate(frame)
            disp2d.paint(img)

        i += 1
        """
        if i == 10:
          with open('map.json', 'w') as f:
            f.write(mapp.serialize())
            exit(0)
        """

