#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
from audioop import avg
from cv2 import threshold
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import imp
import yaml
import time
import __init__ as booger
import os
import numpy as np
from tasks.semantic.modules.CorrelationL import *
from tasks.semantic.modules.MEDROR import *

class User():
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir, split):
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
                                   self.DATA["name"] + '/multi_parser.py')
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
                                      gt=False, 
                                      shuffle_train=False)
    
    self.n_echoes = self.ARCH["dataset"]["sensor"]["n_echoes"]

    # concatenate the encoder and the head
    with torch.no_grad():
        torch.nn.Module.dump_patches = True
        
        self.medror = medror(n_echoes=self.ARCH["dataset"]["sensor"]["n_echoes"])

        self.model = CorrL(self.parser.get_n_classes(), ARCH, n_echoes=self.ARCH["dataset"]["sensor"]["n_echoes"])
        w_dict = torch.load(modeldir + "/SMEDNet",
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
      self.medror.cuda()

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
    print("Mean CNN inference time:{}\t std:{}".format(np.mean(cnn), np.std(cnn)))
    print("Total Frames:{}".format(len(cnn)))
    print("Finished Infering")

    return

  def infer_subset(self, loader, to_orig_fn, cnn):
    self.model.eval()

    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():

      for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, p_z, _, unproj_range, _, _, _, npoints, stack_order) in enumerate(loader):
        p_x = p_x[0, :npoints]
        p_y = p_y[0, :npoints]
        p_z = p_z[0, :npoints]
        unproj_range = unproj_range[0, :npoints]
        path_seq = path_seq[0]
        path_name = path_name[0]

        if self.gpu:
          proj_in = proj_in.cuda()
          p_x = p_x.cuda()
          p_y = p_y.cuda()
          p_z = p_z.cuda()
          stack_order = stack_order.cuda()

        valid_mask = (proj_in[:, 0:self.n_echoes, ...] > 0).int()
        proj_range = torch.clamp(proj_in[:, 0:self.n_echoes, ...], 1.0, 80.0)
        if self.gpu: proj_range = proj_range.cuda()
        
        #compute output
        end = time.time()
        use_medror = False
        if use_medror:
          proj_output = self.medror(proj_in)
        else:
          proj_output = self.model(proj_in)
          proj_output = proj_output * valid_mask
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = time.time() - end
        print("Network seq", path_seq, "scan", path_name, "in", res, "sec")
        end = time.time()
        cnn.append(res)

        proj_output = proj_output.squeeze()
        proj_predictions = torch.zeros_like(proj_output)

        # is it best
        is_best = torch.zeros_like(proj_output)
        is_best_un = is_best.view(is_best.shape[0], is_best.shape[1] * is_best.shape[2])
        proj_output_un = proj_output.clone().view(proj_output.shape[0], proj_output.shape[1] * proj_output.shape[2])
        is_best_un[proj_output_un.argmin(dim=0), torch.arange(is_best_un.size(1))] = 1
        is_best = is_best_un.view(is_best.shape[0], is_best.shape[1], is_best.shape[2]).bool()
        
        # is it strongest
        is_strongest = torch.zeros_like(proj_output)
        is_strongest[0, ...] = 1
        is_strongest = is_strongest.bool()
        
        # is it under threshold
        thresh = -1.00
        under_threshold = (proj_output < thresh)

        # is range different compared to strogest
        is_different = torch.zeros_like(proj_output)
        is_different[proj_range.squeeze()[0, ...] + 1 < proj_range.squeeze()] = 1
        is_different = is_different.bool()
        
        # is 1 if strongest and under threshold
        proj_predictions[is_strongest * under_threshold] = 1

        # is 2 if strongest and over threshold
        proj_predictions[is_strongest * ~under_threshold] = 2

        # is 3 (substitute) if not strongest but better than strongest and has different coordinate than strongest
        proj_predictions[~is_strongest * under_threshold * is_best * is_different] = 3

        # is 4 if discarded
        proj_predictions[~is_strongest * ~is_best] = 4
        proj_predictions[~is_strongest * ~under_threshold] = 4
        proj_predictions[~is_strongest * ~is_different] = 4
        proj_predictions[proj_range.squeeze() is None] = 4

        # roi
        proj_predictions[is_strongest * proj_range.squeeze() > 30] = 1

        # reverse sort predictions for p_z to match
        stack_order = stack_order.squeeze().permute(2, 0, 1)
        unsorted_proj_predictions = torch.gather(input=proj_predictions, dim=0, index=stack_order.argsort(dim=0))

        unproj_predictions = unsorted_proj_predictions[p_z, p_y, p_x]

        # measure elapsed time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()

        # save scan
        # get the first scan in batch and project scan
        pred_np = unproj_predictions.cpu().numpy()
        pred_np = pred_np.reshape((-1)).astype(np.int32)

        # map to original label
        pred_np = to_orig_fn(pred_np)

        # save scan
        path = os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name)
        pred_np.tofile(path)
