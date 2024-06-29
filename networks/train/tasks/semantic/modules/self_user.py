#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

from audioop import avg
from cv2 import threshold
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import imp
import yaml
import time
from PIL import Image
import __init__ as booger
import os
import numpy as np
from tasks.semantic.modules.SCorrelationL import *
from tasks.semantic.modules.SCoordinateL import *
from tasks.semantic.modules.Slide import *

class User():
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir,split):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.logdir = logdir
    self.modeldir = modeldir
    self.split = split

    # get the data
    parserModule = imp.load_source("parserModule",
                                   booger.TRAIN_PATH + '/tasks/semantic/dataset/' +
                                   self.DATA["name"] + '/parser.py')
    self.parser = parserModule.Parser(root=self.datadir,
                                      train_sequences=self.DATA["split"]["train"],
                                      valid_sequences=self.DATA["split"]["valid"],
                                      test_sequences=self.DATA["split"]["test"],
                                      split = self.split,
                                      labels=self.DATA["labels"],
                                      color_map=self.DATA["color_map"],
                                      learning_map=self.DATA["learning_map"],
                                      learning_map_inv=self.DATA["learning_map_inv"],
                                      sensor=self.ARCH["dataset"]["sensor"],
                                      max_points=self.ARCH["dataset"]["max_points"],
                                      batch_size=1,
                                      workers=self.ARCH["train"]["workers"],
                                      gt=True,
                                      shuffle_train=False)

    # concatenate the encoder and the head
    self.slide = False
    with torch.no_grad():
        torch.nn.Module.dump_patches = True

        if not self.slide:
            self.model = SCorrL(self.parser.get_n_classes(), ARCH)
        else:
            self.model = RDNet(self.parser.get_n_classes(), ARCH)
        
        #self.model = nn.DataParallel(self.model)
        w_dict = torch.load(modeldir + "/SMEDNet_valid_best",
                            map_location=lambda storage, loc: storage)
        self.model.load_state_dict(w_dict['state_dict'], strict=True)

    # GPU?
    self.gpu = False
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Infering in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.model.cuda()

  def infer(self):
    cnn = []
    if self.split == None:

        self.infer_subset(loader=self.parser.get_train_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn)

        # do valid set
        self.infer_subset(loader=self.parser.get_valid_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn)
        # do test set
        self.infer_subset(loader=self.parser.get_test_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn)


    elif self.split == 'valid':
        self.infer_subset(loader=self.parser.get_valid_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn)
    elif self.split == 'train':
        self.infer_subset(loader=self.parser.get_train_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn)
    else:
        self.infer_subset(loader=self.parser.get_test_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn)
    print("Mean inference time:{}\t std:{}".format(np.mean(cnn), np.std(cnn)))
    print("Total Frames:{}".format(len(cnn)))
    print("Finished Infering")

    return

  def infer_subset(self, loader, to_orig_fn, cnn):
    
    self.model.eval()
    
    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      #end = time.time()

      for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, unproj_xyz, _, _, npoints) in enumerate(loader):
        # first cut to rela size (batch size one allows it)
        p_x = p_x[0, :npoints]
        p_y = p_y[0, :npoints]
        proj_range = proj_range[0, :npoints]
        unproj_range = unproj_range[0, :npoints]
        unproj_xyz = unproj_xyz[0, :npoints, :]
        path_seq = path_seq[0]
        path_name = path_name[0]

        if self.gpu:
          proj_in = proj_in.cuda()
          p_x = p_x.cuda()
          p_y = p_y.cuda()
          

        #compute output
        valid_mask = (proj_in[:, 0, :, :] != 0).int()
        proj_range = torch.clamp(proj_range, 0.5, 200).cuda().unsqueeze(dim=0)

        end = time.time()
        #compute output
        proj_output = self.model(proj_in) * valid_mask
        res = time.time() - end

        thresh = 0
        proj_argmax = ((proj_output > thresh).int() + 1).squeeze()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        print("Network seq", path_seq, "scan", path_name, "in", res, "sec")
        end = time.time()
        cnn.append(res)

        unproj_argmax = proj_argmax[p_y, p_x]

        # hack to prevent unvalid points to be classified wrongly
        unproj_argmax[torch.all(unproj_xyz == -1.0, axis=1)] = 1

        # measure elapsed time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()

        # save scan
        # get the first scan in batch and project scan
        pred_np = unproj_argmax.cpu().numpy()
        pred_np = pred_np.reshape((-1)).astype(np.int32)

        # map to original label
        pred_np = to_orig_fn(pred_np)

        # save scan
        path = os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name)
        pred_np.tofile(path)
