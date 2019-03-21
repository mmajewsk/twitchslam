from helpers import poseRt, hamming_distance, add_ones
from constants import CULLING_ERR_THRES
from frame import Frame
import time
import numpy as np
import g2o
import json

from optimize_g2o import Optimize
import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s]= %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#from optimize_crappy import optimize

LOCAL_WINDOW = 20
#LOCAL_WINDOW = None

class Point(object):
    # A Point is a 3-D point in the world
    # Each Point is observed in multiple Frames

    def __init__(self, mapp, loc, color, tid=None):
        self.pt = np.array(loc)
        self.frames = []
        self.idxs = []
        self.color = np.copy(color)
        self.id = tid if tid is not None else mapp.add_point(self)

    def homogeneous(self):
        return add_ones(self.pt)

    def orb(self):
        return [f.descriptors[idx] for f,idx in zip(self.frames, self.idxs)]

    def orb_distance(self, descriptors):
        return min([hamming_distance(o, descriptors) for o in self.orb()])

    def delete(self):
        for f,idx in zip(self.frames, self.idxs):
            f.pts[idx] = None
        del self

    def join_frame(self, frame, idx):
        assert frame not in self.frames
        self.frames.append(frame)
        self.idxs.append(idx)

    def leave_frame(self, frame, idx):
        assert frame in self.frames
        del self.frames[idx]
        assert frame not in self.frames

def connect_frame_point(frame: Frame, point: Point, idx):
    point.join_frame(frame, idx)
    frame.join_point(point, idx)

def disconnect_frame_point(frame: Frame, point: Point, idx):
    point.leave_frame(frame, idx)
    frame.leave_point(frame, idx)

class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []
        self.max_frame = 0
        self.max_point = 0

    def serialize(self):
        ret = {}
        ret['points'] = [{'id': p.id, 'pt': p.pt.tolist(), 'color': p.color.tolist()} for p in self.points]
        ret['frames'] = []
        for f in self.frames:
            ret['frames'].append({
                'id': f.id, 'K': f.K.tolist(), 'pose': f.pose.tolist(), 'h': f.h, 'w': f.w,
                'kpus': f.kpus.tolist(), 'descriptors': f.descriptors.tolist(),
                'pts': [p.id if p is not None else -1 for p in f.pts]})
        ret['max_frame'] = self.max_frame
        ret['max_point'] = self.max_point
        return json.dumps(ret)

    def deserialize(self, s):
        ret = json.loads(s)
        self.max_frame = ret['max_frame']
        self.max_point = ret['max_point']
        self.points = []
        self.frames = []

        pids = {}
        for p in ret['points']:
            pp = Point(self, p['pt'], p['color'], p['id'])
            self.points.append(pp)
            pids[p['id']] = pp

        for f in ret['frames']:
            ff = Frame(self, None, f['K'], f['pose'], f['id'])
            ff.w, ff.h = f['w'], f['h']
            ff.key_points = np.array(f['kpus'])
            ff.descriptors = np.array(f['descriptors'])
            ff.pts = [None] * len(ff.key_points)
            for i,p in enumerate(f['pts']):
                if p != -1:
                    ff.pts[i] = pids[p]
            self.frames.append(ff)

    def add_point(self, point):
        ret = self.max_point
        self.max_point += 1
        self.points.append(point)
        return ret

    def add_frame(self, frame):
        ret = self.max_frame
        self.max_frame += 1
        self.frames.append(frame)
        return ret

    def pop_frame(self):
        self.max_point -= 1
        return self.frames.pop()
    # *** optimizer ***

    def connect_observations(self, frame_1, frame_2, idx1, idx2):
        # add new observations if the point is already observed in the previous frame
        # TODO: consider tradeoff doing this before/after search by projection
        for i,idx in enumerate(idx2):
            if frame_2.pts[idx] is not None and frame_1.pts[idx1[i]] is None:
                connect_frame_point(frame_1, frame_2.pts[idx], idx1[i])

    def optimise_pose(self, pose, frame_1):
        #@TODO #mmajewsk explain. This code seems to not be doing anything
        # pose optimization
        if pose is None:
            #print(frame_1.pose)
            pose_opt = self.optimize(local_window=1, fix_points=True)
            logger.info("Pose:     %f" % pose_opt)
            #print(frame_1.pose)
        else:
            # have ground truth for pose
            frame_1.pose = pose
        return frame_1

    def search_by_projection(self, frame_1, K, W, H):
        # search by projection
        sbp_pts_count = 0
        if len(self.points) > 0:
            # project *all* the map points into the current frame
            map_points = np.array([p.homogeneous() for p in self.points])
            projs = np.dot(np.dot(K, frame_1.pose[:3]), map_points.T).T
            projs = projs[:, 0:2] / projs[:, 2:]

            # only the points that fit in the frame
            good_pts = (projs[:, 0] > 0) & (projs[:, 0] < W) & \
                       (projs[:, 1] > 0) & (projs[:, 1] < H)

            for i, p in enumerate(self.points):
                if not good_pts[i]:
                    # point not visible in frame
                    continue
                if frame_1 in p.frames:
                    # we already matched this map point to this frame
                    # TODO: understand this better
                    continue
                for m_idx in frame_1.kd.query_ball_point(projs[i], 2):
                    # if point unmatched
                    if frame_1.pts[m_idx] is None:
                        b_dist = p.orb_distance(frame_1.descriptors[m_idx])
                        # if any descriptors within 64
                        if b_dist < 64.0:
                            connect_frame_point(frame_1, p,  m_idx)
                            sbp_pts_count += 1
                            break
        return sbp_pts_count

    def optimize(self, local_window=LOCAL_WINDOW, fix_points=False, verbose=False, rounds=50, reset_map_optimiser=True):
        map_optimizer = Optimize(fix_points=fix_points, verbose=verbose)
        err = map_optimizer.optimise(local_window, self.frames, self.points, rounds=rounds)

        # prune points
        culled_pt_count = 0
        for p in self.points:
            # <= 4 match point that's old
            old_point = len(p.frames) <= 4 and p.frames[-1].id+7 < self.max_frame

            # compute reprojection error
            errs = []
            for f,idx in zip(p.frames, p.idxs):
                uv = f.kps[idx]
                proj = np.dot(f.pose[:3], p.homogeneous())
                proj = proj[0:2] / proj[2]
                errs.append(np.linalg.norm(proj-uv))

            # cull
            if old_point or np.mean(errs) > CULLING_ERR_THRES:
                culled_pt_count += 1
                self.points.remove(p)
                p.delete()
        print("Culled:   %d points" % (culled_pt_count))

        return err

