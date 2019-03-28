#!/usr/bin/env python3

import sys

sys.path.append("lib/macosx")
sys.path.append("lib/linux")
import matplotlib.pyplot as plt
import time
from frame import Frame, NoFrameMatchError, match_frames
import numpy as np
from pointmap import Map, Point, connect_frame_point
from helpers import triangulate, add_ones
import logging
import cv2
np.set_printoptions(suppress=True)

logging.basicConfig(format='%(asctime)s [%(levelname)s]= %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Tracker:
    def __init__(self, mapp):
        self.mapp = mapp
        self.frame_1 = None
        self.frame_2 = None

    def image_to_track(self, img, verts, K):
        self.frame_2 = self.frame_1
        self.frame_1 = Frame(img, K, verts=verts)

    def track(self):
        return match_frames(self.frame_1, self.frame_2)

    def make_initial_pose(self, rotation_matrix):
        if self.frame_1.id < 5 or True:
            # get initial positions from fundamental matrix
            # apply the rotation matrix to the rotation matrix from previous round
            self.frame_1.pose = np.dot(rotation_matrix, self.frame_2.pose)
        else:
            # kinematic model (not used)
            velocity = np.dot(self.frame_2.pose, np.linalg.inv(self.mapp.frames[-3].pose))
            self.frame_1.pose = np.dot(velocity, self.frame_2.pose)
        return self.frame_1

class SLAM(object):
    def __init__(self, W, H, K):
        # main classes
        self.mapp = Map()
        # params
        self.W, self.H = W, H
        self.K = K
        self.distances = []
        self.tracker = Tracker(self.mapp)


    def triangulate(self, frame_1, frame_2, idx1, idx2):
        # triangulate the points we don't have matches for
        good_pts4d = np.array([frame_1.pts[i] is None for i in idx1])
        # do triangulation in global frame
        pts4d = triangulate(frame_1.pose, frame_2.pose, frame_1.kps[idx1], frame_2.kps[idx2])
        good_pts4d &= np.abs(pts4d[:, 3]) != 0
        pts4d /= pts4d[:, 3:]       # homogeneous 3-D coords
        return pts4d, good_pts4d

    def match_points(self, pts4d, good_pts4d, frame_1, frame_2, idx1, idx2, img, new_pts_count):
        for i,p in enumerate(pts4d):
            if not good_pts4d[i]:
                continue

            # check parallax is large enough
            # TODO: learn what parallax means
            """
            r1 = np.dot(frame_1.pose[:3, :3], add_ones(frame_1.kps[idx1[i]]))
            r2 = np.dot(frame_2.pose[:3, :3], add_ones(frame_2.kps[idx2[i]]))
            parallax = r1.dot(r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
            if parallax >= 0.9998:
              continue
            """

            # check points are in front of both cameras
            pl1 = np.dot(frame_1.pose, p)
            pl2 = np.dot(frame_2.pose, p)
            if pl1[2] < 0 or pl2[2] < 0:
                continue

            # reproject
            pp1 = np.dot(self.K, pl1[:3])
            pp2 = np.dot(self.K, pl2[:3])

            # check reprojection error
            pp1 = (pp1[0:2] / pp1[2]) - frame_1.key_points[idx1[i]]
            pp2 = (pp2[0:2] / pp2[2]) - frame_2.key_points[idx2[i]]
            pp1 = np.sum(pp1**2)
            pp2 = np.sum(pp2**2)
            if pp1 > 2 or pp2 > 2:
                continue

            # add the point
            try:
                color = img[int(round(frame_1.key_points[idx1[i],1])), int(round(frame_1.key_points[idx1[i],0]))]
            except IndexError:
                color = (255,0,0)
            pt = Point(self.mapp, p[0:3], color)
            connect_frame_point(frame_2, pt, idx2[i])
            connect_frame_point(frame_1, pt, idx1[i])
            new_pts_count += 1

    def reoptimize(self, idx1, idx2, img):
        sbp_pts_count = self.mapp.search_by_projection(self.tracker.frame_1, self.K, self.W, self.H)
        # adding new points to the map from pairwise matches
        new_pts_count = 0
        pts4d, good_pts4d = self.triangulate( self.tracker.frame_1, self.tracker.frame_2, idx1, idx2)
        self.match_points(pts4d, good_pts4d, self.tracker.frame_1, self.tracker.frame_2, idx1, idx2, img, new_pts_count)
        return new_pts_count, sbp_pts_count


    def process_frame(self, img, pose=None, verts=None):
        assert img.shape[0:2] == (self.H, self.W)
        start_time = time.time()
        self.tracker.image_to_track(img, verts, self.K)
        self.tracker.frame_1.id = self.mapp.add_frame(self.tracker.frame_1)
        if self.tracker.frame_1.id == 0:
            return
        try:
            idx1, idx2, rotation_matrix = self.tracker.track()
            frame_1 = self.tracker.make_initial_pose( rotation_matrix)
            self.mapp.connect_observations(self.tracker.frame_1, self.tracker.frame_2, idx1, idx2)
            frame_1 = self.mapp.optimise_pose(pose, frame_1)
            new_pts_count, sbp_pts_count = self.reoptimize(idx1, idx2, img)
            # optimize the map
            if frame_1.id >= 4 and frame_1.id%5 == 0:
                err = self.mapp.optimize() #verbose=True)
                logger.info("Optimize: %f units of error" % err)

        except NoFrameMatchError as e:
            self.mapp.pop_frame()
            raise e
        # pose_matrix = np.linalg.inv(frame_1.pose)
        # previous_pose_matrix = np.linalg.inv(frame_2.pose)
        # distance = pose_matrix[:3,3] - previous_pose_matrix[:3,3]
        # if frame_1.id > 5:
        #     self.distances.append(distance)
        #     plt.plot(self.distances)
        #     plt.show()
        movement = cv2.absdiff(frame_1.img, self.tracker.frame_2.img)
        logger.info("Adding:   %d new points, %d search by projection" % (new_pts_count, sbp_pts_count))
        logger.info("Map:      %d points, %d frames" % (len(self.mapp.points), len(self.mapp.frames)))
        logger.info("Time:     %.2f ms" % ((time.time()-start_time)*1000.0))
        return dict(movement=movement)


