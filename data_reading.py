import pathlib
import cv2
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s]= %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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