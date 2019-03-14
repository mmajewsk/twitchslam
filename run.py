from slam import SLAM
import cv2
import numpy as np
from display import Display2D, Display3D, annotate
import argparse
import pathlib
from frame import NoFrameMatchError
import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s]= %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class StreamVideo:
    def __init__(self, source_path):
        self.vid_cap = cv2.VideoCapture(str(source_path))

    def is_next(self):
        return self.vid_cap.isOpened()

    def get_next(self):
        return self.vid_cap.read()

    def set_seek(self, seek):
        self.vid_cap.set(cv2.CAP_PROP_POS_FRAMES, int(seek))

    def params(self):
        W = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        CNT = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return W, H, CNT

class StreamFolder:
    def __init__(self, source_path):
        self.folder = source_path
        self.files = [file for file in source_path.iterdir() if file.suffix == '.jpg']
        self.files = sorted(self.files)
        self.iterator = 0

    def set_seek(self, seek):
        self.iterator = seek

    def params(self):
        img = cv2.imread(str(self.files[0]))
        print(img.shape)
        return img.shape[1], img.shape[0], len(self.files)

    def is_next(self):
        return self.iterator < len(self.files)

    def get_next(self):
        self.iterator += 1
        p = str(self.files[self.iterator-1])
        logger.info(p)
        return 1, cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)

class ImageStream:
    def is_folder(self, file_path : pathlib.Path):
        return	file_path.is_dir() and  any([file.suffix == '.jpg' for file in file_path.iterdir()])

    def is_video(self, file_path: pathlib.Path):
        return file_path.suffix in ['.mp4']

    def __init__(self, source : str):
        source_path = pathlib.Path(source)
        if self.is_video(source_path):
            self.ss = StreamVideo(source_path)
        elif self.is_folder(source_path):
            self.ss = StreamFolder(source_path)
        else:
            raise ValueError()



    def create_camera_params(self, F):
        W, H, CNT = self.ss.params()

        if W > 1024:
            downscale = 1024.0 / W
            F *= downscale
            H = int(H * downscale)
            W = 1024

        # camera intrinsics
        K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])
        Kinv = np.linalg.inv(K)

        logger.info("using camera %dx%d with F %f" % (W, H, F))
        return W, H, K, CNT


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

