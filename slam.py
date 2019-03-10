#!/usr/bin/env python3

import sys

sys.path.append("lib/macosx")
sys.path.append("lib/linux")

import time
from frame import Frame, match_frames
import numpy as np
from pointmap import Map, Point
from helpers import triangulate, add_ones
import logging

np.set_printoptions(suppress=True)

logging.basicConfig(format='%(asctime)s [%(levelname)s]= %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



class SLAM(object):
    def __init__(self, W, H, K):
        # main classes
        self.mapp = Map()

        # params
        self.W, self.H = W, H
        self.K = K

    def add_observations(self, f1, f2, idx1, idx2):
        # add new observations if the point is already observed in the previous frame
        # TODO: consider tradeoff doing this before/after search by projection
        # @TODO #mmajewsk explain
        for i,idx in enumerate(idx2):
            if f2.pts[idx] is not None and f1.pts[idx1[i]] is None:
                f2.pts[idx].add_observation(f1, idx1[i])
        return f2

    def make_initial_pose(self, frame, Rt, f1, f2):
        # @TODO #mmajewsk explain
        if frame.id < 5 or True:
            # get initial positions from fundamental matrix
            f1.pose = np.dot(Rt, f2.pose)
        else:
            # kinematic model (not used)
            velocity = np.dot(f2.pose, np.linalg.inv(self.mapp.frames[-3].pose))
            f1.pose = np.dot(velocity, f2.pose)
        return f1

    def optimise_pose(self, pose, f1):
        #@TODO #mmajewsk explain. This code seems to not be doing anything
        # pose optimization
        if pose is None:
            #print(f1.pose)
            pose_opt = self.mapp.optimize(local_window=1, fix_points=True)
            logger.info("Pose:     %f" % pose_opt)
            #print(f1.pose)
        else:
            # have ground truth for pose
            f1.pose = pose
        return f1

    def search_by_projection(self, f1):
        # search by projection
        sbp_pts_count = 0
        if len(self.mapp.points) > 0:
            # project *all* the map points into the current frame
            map_points = np.array([p.homogeneous() for p in self.mapp.points])
            projs = np.dot(np.dot(self.K, f1.pose[:3]), map_points.T).T
            projs = projs[:, 0:2] / projs[:, 2:]

            # only the points that fit in the frame
            good_pts = (projs[:, 0] > 0) & (projs[:, 0] < self.W) & \
                       (projs[:, 1] > 0) & (projs[:, 1] < self.H)

            for i, p in enumerate(self.mapp.points):
                if not good_pts[i]:
                    # point not visible in frame
                    continue
                if f1 in p.frames:
                    # we already matched this map point to this frame
                    # TODO: understand this better
                    continue
                for m_idx in f1.kd.query_ball_point(projs[i], 2):
                    # if point unmatched
                    if f1.pts[m_idx] is None:
                        b_dist = p.orb_distance(f1.des[m_idx])
                        # if any descriptors within 64
                        if b_dist < 64.0:
                            p.add_observation(f1, m_idx)
                            sbp_pts_count += 1
                            break
        return sbp_pts_count

    def triangulate(self, f1, f2, idx1, idx2):
        # triangulate the points we don't have matches for
        good_pts4d = np.array([f1.pts[i] is None for i in idx1])

        # do triangulation in global frame
        pts4d = triangulate(f1.pose, f2.pose, f1.kps[idx1], f2.kps[idx2])
        good_pts4d &= np.abs(pts4d[:, 3]) != 0
        pts4d /= pts4d[:, 3:]       # homogeneous 3-D coords
        return pts4d, good_pts4d

    def match_points(self, pts4d, good_pts4d, f1, f2, idx1, idx2, img, new_pts_count):
        for i,p in enumerate(pts4d):
            if not good_pts4d[i]:
                continue

            # check parallax is large enough
            # TODO: learn what parallax means
            """
            r1 = np.dot(f1.pose[:3, :3], add_ones(f1.kps[idx1[i]]))
            r2 = np.dot(f2.pose[:3, :3], add_ones(f2.kps[idx2[i]]))
            parallax = r1.dot(r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
            if parallax >= 0.9998:
              continue
            """

            # check points are in front of both cameras
            pl1 = np.dot(f1.pose, p)
            pl2 = np.dot(f2.pose, p)
            if pl1[2] < 0 or pl2[2] < 0:
                continue

            # reproject
            pp1 = np.dot(self.K, pl1[:3])
            pp2 = np.dot(self.K, pl2[:3])

            # check reprojection error
            pp1 = (pp1[0:2] / pp1[2]) - f1.kpus[idx1[i]]
            pp2 = (pp2[0:2] / pp2[2]) - f2.kpus[idx2[i]]
            pp1 = np.sum(pp1**2)
            pp2 = np.sum(pp2**2)
            if pp1 > 2 or pp2 > 2:
                continue

            # add the point
            try:
                color = img[int(round(f1.kpus[idx1[i],1])), int(round(f1.kpus[idx1[i],0]))]
            except IndexError:
                color = (255,0,0)
            pt = Point(self.mapp, p[0:3], color)
            pt.add_observation(f2, idx2[i])
            pt.add_observation(f1, idx1[i])
            new_pts_count += 1


    def _profiled_process_frame(self, frame, img, pose=None, verts=None):
        f1 = self.mapp.frames[-1]
        f2 = self.mapp.frames[-2]
        idx1, idx2, Rt = match_frames(f1, f2)
        f2 = self.add_observations(f1,f2,idx1,idx2)
        f1 = self.make_initial_pose( frame, Rt, f1, f2)
        f1 = self.optimise_pose(pose, f1)
        sbp_pts_count = self.search_by_projection(f1)
        # adding new points to the map from pairwise matches
        new_pts_count = 0
        pts4d, good_pts4d = self.triangulate(f1, f2, idx1, idx2)
        self.match_points(pts4d, good_pts4d, f1, f2, idx1, idx2, img, new_pts_count)
        # optimize the map
        if frame.id >= 4 and frame.id%5 == 0:
            err = self.mapp.optimize() #verbose=True)
            logger.info("Optimize: %f units of error" % err)
        return new_pts_count, sbp_pts_count, f1

    def process_frame(self, img, pose=None, verts=None):
        assert img.shape[0:2] == (self.H, self.W)
        start_time = time.time()
        frame = Frame(self.mapp, img, self.K, verts=verts)
        if frame.id == 0:
            return
        new_pts_count, sbp_pts_count, f1 = self._profiled_process_frame(frame, img, pose, verts)
        logger.info("Adding:   %d new points, %d search by projection" % (new_pts_count, sbp_pts_count))
        logger.info("Map:      %d points, %d frames" % (len(self.mapp.points), len(self.mapp.frames)))
        logger.info("Time:     %.2f ms" % ((time.time()-start_time)*1000.0))
        logger.info(np.linalg.inv(f1.pose))


