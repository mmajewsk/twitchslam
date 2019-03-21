from data_reading import ImageStream
from slam import SLAM
import cv2
import numpy as np
from display import Display2D, Display3D, annotate
import argparse
from frame import NoFrameMatchError
import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s]= %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(stream):
    i = 0
    prev_gray = None
    while stream.ss.is_next():
        ret, image = stream.ss.get_next()
        image = cv2.resize(image, (W, H))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mov = None
        if prev_gray is not None:
            mov = cv2.absdiff(gray,prev_gray)
        if mov is not None  and np.sum(mov) < 100 :
            prev_gray = gray
            logger.debug("Skipping no movement")
            continue
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        logger.debug("Blur: {}".format(blur))
        logger.info("\n*** image %d/%d ***" % (i, CNT))
        if ret == True:
            try:
                params = slam.process_frame(image, None if gt_pose is None else np.linalg.inv(gt_pose[i]))
            except NoFrameMatchError:
                continue

        else:
            break

        # 3-D display
        if disp3d is not None:
            disp3d.paint(slam.mapp)

        if disp2d is not None:
            img = annotate(slam.mapp.frames[-1],image)
            disp2d.paint(img)
            if params and 'movement' in params.keys():
                if prev_gray is not None:
                    disp2d.add_text(str(np.sum(mov)))
                    cv2.imshow('movement', mov)
                cv2.waitKey(1)
        i += 1
        prev_gray = gray
        """
        if i == 10:
          with open('map.json', 'w') as f:
            f.write(mapp.serialize())
            exit(0)
        """

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", type=str, help="path to the video")
    parser.add_argument("pose_path",  nargs="?", type=str, help="pose file path", default=None)
    parser.add_argument("--headless", type=bool, default=None)
    parser.add_argument("-f", type=int, default=525)
    parser.add_argument("--seek", type=int, default=False)
    args = parser.parse_args()
    stream = ImageStream(args.source)
    W, H, K, CNT = stream.create_camera_params(float(args.f))
    if args.seek:
        stream.ss.set_seek(args.seek)
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

    main(stream)
