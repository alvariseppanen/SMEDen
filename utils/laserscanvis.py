#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

from curses import color_content
from traceback import print_tb
from turtle import bgcolor
import vispy
from vispy.scene import visuals, SceneCanvas
from vispy.visuals import markers
import numpy as np
from matplotlib import pyplot as plt
from laserscan import LaserScan, SemLaserScan
import imageio  # for animation

save_animation = False

class LaserScanVis:
  """Class that creates and handles a visualizer for a pointcloud"""

  def __init__(self, scan, scan_names, label_names, offset=0, multi_echo=True,
               semantics=True, instances=False):
    self.scan = scan
    self.scan_names = scan_names
    self.label_names = label_names
    self.offset = offset
    self.multi_echo = multi_echo
    self.total = len(self.scan_names)
    self.semantics = semantics
    self.instances = instances

    if save_animation:
      self.writer = imageio.get_writer('~/Desktop/animation.gif')  # for animation

    # sanity check
    if not self.semantics and self.instances:
      print("Instances are only allowed in when semantics=True")
      raise ValueError

    self.reset()
    self.update_scan()

  def reset(self):
    """ Reset. """
    # last key press (it should have a mutex, but visualization is not
    # safety critical, so let's do things wrong)
    self.action = "no"  # no, next, back, quit are the possibilities

    self.point_size = 8
    # new canvas prepared for visualizing data
    self.canvas = SceneCanvas(size=(1600, 800), keys='interactive', show=True)
    # interface (n next, b back, q quit)
    self.canvas.events.key_press.connect(self.key_press)
    self.canvas.events.draw.connect(self.draw)
    # grid
    self.grid = self.canvas.central_widget.add_grid()

    # laserscan part
    self.scan_view = vispy.scene.widgets.ViewBox(
        border_color='black', parent=self.canvas.scene, bgcolor='white')
    self.grid.add_widget(self.scan_view, 0, 0)
    self.scan_vis = visuals.Markers()
    self.scan_view.camera = 'turntable'
    elev, azim, scale = 35, -70, 25
    self.scan_view.camera.elevation = elev
    self.scan_view.camera.azimuth = azim
    self.scan_view.camera.scale_factor = scale
    self.scan_view.add(self.scan_vis)
    visuals.XYZAxis(parent=self.scan_view.scene, width=3)
    # add semantics
    if self.semantics:
      print("Using semantics in visualizer")
      self.sem_view = vispy.scene.widgets.ViewBox(
          border_color='black', parent=self.canvas.scene, bgcolor='white')
      self.grid.add_widget(self.sem_view, 0, 1)
      self.sem_vis = visuals.Markers()
      self.sem_view.camera = 'turntable'
      self.sem_view.camera.elevation = elev
      self.sem_view.camera.azimuth = azim
      self.sem_view.camera.scale_factor = scale
      self.sem_view.add(self.sem_vis)
      visuals.XYZAxis(parent=self.sem_view.scene, width=3)

    if self.instances:
      print("Using instances in visualizer")
      self.inst_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.canvas.scene)
      self.grid.add_widget(self.inst_view, 0, 2)
      self.inst_vis = visuals.Markers()
      self.inst_view.camera = 'turntable'
      self.inst_view.add(self.inst_vis)
      visuals.XYZAxis(parent=self.inst_view.scene)

    # img canvas size
    self.multiplier = 1
    self.canvas_W = 1024
    self.canvas_H = 64
    if self.semantics:
      self.multiplier += 1
    if self.instances:
      self.multiplier += 1

    # new canvas for img
    self.img_canvas = SceneCanvas(keys='interactive', show=True,
                                  size=(self.canvas_W, self.canvas_H * self.multiplier))
    # grid
    self.img_grid = self.img_canvas.central_widget.add_grid()
    # interface (n next, b back, q quit, very simple)
    self.img_canvas.events.key_press.connect(self.key_press)
    self.img_canvas.events.draw.connect(self.draw)

    # add a view for the depth
    self.img_view = vispy.scene.widgets.ViewBox(
        border_color='white', parent=self.img_canvas.scene)
    self.img_grid.add_widget(self.img_view, 0, 0)
    self.img_vis = visuals.Image(cmap='viridis')
    self.img_view.add(self.img_vis)

    # add semantics
    if self.semantics:
      self.sem_img_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.img_canvas.scene)
      self.img_grid.add_widget(self.sem_img_view, 1, 0)
      self.sem_img_vis = visuals.Image(cmap='viridis')
      self.sem_img_view.add(self.sem_img_vis)

    # add instances
    if self.instances:
      self.inst_img_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.img_canvas.scene)
      self.img_grid.add_widget(self.inst_img_view, 2, 0)
      self.inst_img_vis = visuals.Image(cmap='viridis')
      self.inst_img_view.add(self.inst_img_vis)

  def get_mpl_colormap(self, cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0

  def update_scan(self):
    # first open data
    self.scan.open_scan(self.scan_names[self.offset])
    if self.semantics:
      self.scan.open_label(self.label_names[self.offset])
      self.scan.colorize()

    # then change names
    title = "scan " + str(self.offset)
    self.canvas.title = title
    self.img_canvas.title = title

    # then do all the point cloud stuff

    # plot scan
    power = 16
    range_data = np.clip(np.copy(self.scan.points[:,2]), -2, 2)
    viridis_range = ((range_data - range_data.min()) /
                     (range_data.max() - range_data.min()) *
                     255).astype(np.uint8)
    viridis_map = self.get_mpl_colormap("viridis")
    viridis_colors = viridis_map[viridis_range]

    input_points = self.scan.points.copy()
    
    if self.multi_echo and self.semantics:
      input_points[self.scan.sem_label > 1] = None

    self.scan_vis.set_data(input_points,
                           face_color=viridis_colors[..., ::-1],
                           edge_color=viridis_colors[..., ::-1],
                           size=self.point_size, edge_width=1.0)
    
    self.scan_vis.antialias = 0

    # plot semantics
    if self.semantics:

      if self.multi_echo:
        valid_points = self.scan.points.copy()
        valid_points[self.scan.sem_label != 0] = None # remove non-valid points
        substitute_points = self.scan.points.copy()
        substitute_points[self.scan.sem_label != 2] = None
        self.sem_vis.set_data(np.vstack((substitute_points, valid_points)),
                            face_color=np.vstack((self.scan.sem_label_color[..., ::-1], viridis_colors[..., ::-1])),
                            edge_color=np.vstack((self.scan.sem_label_color[..., ::-1], viridis_colors[..., ::-1])),
                            size=self.point_size, edge_width=1.0)
      else:
        self.scan.points[self.scan.sem_label_color == 1.0] = None # remove non-valid points
        self.sem_vis.set_data(self.scan.points,
                            face_color=viridis_colors[..., ::-1],
                            edge_color=viridis_colors[..., ::-1],
                            size=self.point_size, edge_width=1.0)
      
      self.sem_vis.antialias = 0

    # plot instances
    if self.instances:
      self.inst_vis.set_data(self.scan.points,
                             face_color=self.scan.inst_label_color[..., ::-1],
                             edge_color=self.scan.inst_label_color[..., ::-1],
                             size=1)

    # now do all the range image stuff
    # plot range image
    data = np.copy(self.scan.proj_range)
    data[data > 0] = data[data > 0]**(1 / power)
    data[data < 0] = data[data > 0].min()
    data = (data - data[data > 0].min()) / \
        (data.max() - data[data > 0].min())
    self.img_vis.set_data(data)
    self.img_vis.update()

    if save_animation:
      im = self.canvas.render(alpha=True)  # for animation
      self.writer.append_data(im)  # for animation

    if self.semantics:
      self.sem_img_vis.set_data(self.scan.proj_sem_color[..., ::-1])
      self.sem_img_vis.update()

    if self.instances:
      self.inst_img_vis.set_data(self.scan.proj_inst_color[..., ::-1])
      self.inst_img_vis.update()

  # interface
  def key_press(self, event):
    self.canvas.events.key_press.block()
    self.img_canvas.events.key_press.block()
    if event.key == 'N':
      self.offset += 1
      if self.offset >= self.total:
        self.offset = 0
      self.update_scan()
    elif event.key == 'B':
      self.offset -= 1
      if self.offset < 0:
        self.offset = self.total - 1
      self.update_scan()
    elif event.key == 'Q' or event.key == 'Escape':
      if save_animation:
        self.writer.close() # for animation
      self.destroy()

  def draw(self, event):
    if self.canvas.events.key_press.blocked():
      self.canvas.events.key_press.unblock()
    if self.img_canvas.events.key_press.blocked():
      self.img_canvas.events.key_press.unblock()

  def destroy(self):
    # destroy the visualization
    self.canvas.close()
    self.img_canvas.close()
    vispy.app.quit()

  def run(self):
    vispy.app.run()
