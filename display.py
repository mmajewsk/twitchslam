import pygame
from pygame.locals import DOUBLEBUF
from frame import Frame
import numpy as np
import cv2
from helpers import myjet

def annotate(frame: Frame, img : np.ndarray) -> np.ndarray:
        # paint annotations on the image
        for i1 in range(len(frame.kpus)):
            u1, v1 = int(round(frame.kpus[i1][0])), int(round(frame.kpus[i1][1]))
            if frame.pts[i1] is not None:
                if len(frame.pts[i1].frames) >= 5:
                    cv2.circle(img, (u1, v1), color=(0,255,0), radius=3)
                else:
                    cv2.circle(img, (u1, v1), color=(0,128,0), radius=3)
                # draw the trail
                pts = []
                lfid = None
                for f, idx in zip(frame.pts[i1].frames[-9:][::-1], frame.pts[i1].idxs[-9:][::-1]):
                    if lfid is not None and lfid-1 != f.id:
                        break
                    pts.append(tuple(map(lambda x: int(round(x)), f.kpus[idx])))
                    lfid = f.id
                if len(pts) >= 2:
                    cv2.polylines(img, np.array([pts], dtype=np.int32), False, myjet[len(pts)]*255, thickness=1, lineType=16)
            else:
                cv2.circle(img, (u1, v1), color=(0,0,0), radius=3)
        return img


class Display2D(object):
  def __init__(self, W, H):
    pygame.init()
    self.screen = pygame.display.set_mode((W, H), DOUBLEBUF)
    self.surface = pygame.Surface(self.screen.get_size()).convert()

  def paint(self, img):
    # junk
    for event in pygame.event.get():
      pass

    # draw
    #pygame.surfarray.blit_array(self.surface, img.swapaxes(0,1)[:, :, [2,1,0]])

    # RGB, not BGR (might have to switch in twitchslam)
    pygame.surfarray.blit_array(self.surface, img.swapaxes(0,1)[:, :, [0,1,2]])
    self.screen.blit(self.surface, (0,0))

    # blit
    pygame.display.flip()

from multiprocessing import Process, Queue
import pangolin
import OpenGL.GL as gl
import numpy as np

class Display3D(object):
  def __init__(self):
    self.state = None
    self.q = Queue()
    self.vp = Process(target=self.viewer_thread, args=(self.q,))
    self.vp.daemon = True
    self.vp.start()

  def viewer_thread(self, q):
    self.viewer_init(1024, 768)
    while 1:
      self.viewer_refresh(q)

  def viewer_init(self, w, h):
    pangolin.CreateWindowAndBind('Map Viewer', w, h)
    gl.glEnable(gl.GL_DEPTH_TEST)

    self.scam = pangolin.OpenGlRenderState(
      pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000),
      pangolin.ModelViewLookAt(0, -10, -8,
                               0, 0, 0,
                               0, -1, 0))
    self.handler = pangolin.Handler3D(self.scam)

    # Create Interactive View in window
    self.dcam = pangolin.CreateDisplay()
    self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, w/h)
    self.dcam.SetHandler(self.handler)
    # hack to avoid small Pangolin, no idea why it's *2
    self.dcam.Resize(pangolin.Viewport(0,0,w*2,h*2))
    self.dcam.Activate()


  def viewer_refresh(self, q):
    while not q.empty():
      self.state = q.get()

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    self.dcam.Activate(self.scam)

    if self.state is not None:
      if self.state[0].shape[0] >= 2:
        # draw poses
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawCameras(self.state[0][:-1])

      if self.state[0].shape[0] >= 1:
        # draw current pose as yellow
        gl.glColor3f(1.0, 1.0, 0.0)
        pangolin.DrawCameras(self.state[0][-1:])

      if self.state[1].shape[0] != 0:
        # draw keypoints
        gl.glPointSize(5)
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawPoints(self.state[1], self.state[2])

    pangolin.FinishFrame()

  def paint(self, mapp):
    if self.q is None:
      return

    poses, pts, colors = [], [], []
    for f in mapp.frames:
      # invert pose for display only
      poses.append(np.linalg.inv(f.pose))
    for p in mapp.points:
      pts.append(p.pt)
      colors.append(p.color)
    self.q.put((np.array(poses), np.array(pts), np.array(colors)/256.0))



