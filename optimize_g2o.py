import g2o
import numpy as np
from helpers import poseRt

class Optimize:
	def prepare_optimiser(self, verbose):

		# create g2o optimizer
		self.opt = g2o.SparseOptimizer()
		solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
		solver = g2o.OptimizationAlgorithmLevenberg(solver)
		self.opt.set_algorithm(solver)

		# add normalized camera
		cam = g2o.CameraParameters(1.0, (0.0, 0.0), 0)
		cam.set_id(0)
		self.opt.add_parameter(cam)

		self.robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))
		self.graph_frames, self.graph_points = {}, {}
		if verbose:
			self.opt.set_verbose(True)

	def add_frames(self, frames, local_frames):
		# add frames to graph
		for f in (local_frames if self.fix_points else frames):
			pose = f.pose
			se3 = g2o.SE3Quat(pose[0:3, 0:3], pose[0:3, 3])
			v_se3 = g2o.VertexSE3Expmap()
			v_se3.set_estimate(se3)

			v_se3.set_id(f.id * 2)
			v_se3.set_fixed(f.id <= 1 or f not in local_frames)
			#v_se3.set_fixed(f.id != 0)
			self.opt.add_vertex(v_se3)

			# confirm pose correctness
			est = v_se3.estimate()
			assert np.allclose(pose[0:3, 0:3], est.rotation().matrix())
			assert np.allclose(pose[0:3, 3], est.translation())

			self.graph_frames[f] = v_se3

	def add_points(self, points, local_frames):
		# add points to frames
		for p in points:
			if not any([f in local_frames for f in p.frames]):
				continue

			pt = g2o.VertexSBAPointXYZ()
			pt.set_id(p.id * 2 + 1)
			pt.set_estimate(p.pt[0:3])
			pt.set_marginalized(True)
			pt.set_fixed(self.fix_points)
			self.opt.add_vertex(pt)
			self.graph_points[p] = pt
			# add edges
			for f, idx in zip(p.frames, p.idxs):
				if f not in self.graph_frames:
					continue
				edge = g2o.EdgeProjectXYZ2UV()
				edge.set_parameter_id(0, 0)
				edge.set_vertex(0, pt)
				edge.set_vertex(1, self.graph_frames[f])
				edge.set_measurement(f.kps[idx])
				edge.set_information(np.eye(2))
				edge.set_robust_kernel(self.robust_kernel)
				self.opt.add_edge(edge)

	def __init__(self, verbose, fix_points):
		self.prepare_optimiser(verbose)
		self.fix_points = fix_points

	def optimise(self, local_window, frames, points, rounds=50):
		if local_window is None:
			local_frames = frames
		else:
			local_frames = frames[-local_window:]
		self.add_frames(frames, local_frames)
		self.add_points(points, local_frames)
		self.opt.initialize_optimization()
		self.opt.optimize(rounds)
		self.update_values()
		return self.opt.active_chi2()

	def update_values(self ):
		# put frames back
		for f in self.graph_frames:
			est = self.graph_frames[f].estimate()
			R = est.rotation().matrix()
			t = est.translation()
			f.pose = poseRt(R, t)

		# put points back
		if not self.fix_points:
			for p in self.graph_points:
				p.pt = np.array(self.graph_points[p].estimate())

