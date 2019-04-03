import cv2
import math
import numpy as np
import csv
import os
import scipy.io as sio
import h5py
import glob
import json
import pickle


class CAMPUSVideoReader:
    def __init__(self, video_dir):
        self.PrevFrame = -1
        self.Video = cv2.VideoCapture(video_dir, cv2.CAP_FFMPEG)

    def getFrame(self, iFrame):
        current_frame = iFrame

        assert self.Video.get(cv2.CAP_PROP_POS_FRAMES) == current_frame, 'Frame position error'
        result, img = self.Video.read()
        img = img[:, :, ::-1]  # bgr to rgb
        # Update
        self.PrevFrame = current_frame
        return img


def pose2bb(pose):
    renderThreshold = 0.05
    ref_pose = np.array([[0., 0.],  # nose
                         [0., 23.],  # neck
                         [28., 23.],  # rshoulder
                         [39., 66.],  # relbow
                         [45., 108.],  # rwrist
                         [-28., 23.],  # lshoulder
                         [-39., 66.],  # lelbow
                         [-45., 108.],  # lwrist
                         [20., 106.],  # rhip
                         [20., 169.],  # rknee
                         [20., 231.],  # rankle
                         [-20., 106.],  # lhip
                         [-20., 169.],  # lknee
                         [-20., 231.],  # lankle
                         [5., -7.],  # reye
                         [11., -8.],  # rear
                         [-5., -7.],  # leye
                         [-11., -8.],  # lear
                         ])

    # Template bounding box   
    ref_bb = np.array([[-50., -15.],  # left top
                       [50., 240.]])  # right bottom

    pose = np.reshape(pose, (18, 3))
    valid = np.logical_and(np.logical_and(pose[:, 0] != 0, pose[:, 1] != 0), pose[:, 2] >= renderThreshold)

    if np.sum(valid) < 2:
        bb = np.array([0, 0, 0, 0])
        print('got an invalid box')
        return bb

    points_det = pose[valid, 0:2]
    points_reference = ref_pose[valid, :]

    # 1a) Compute minimum enclosing rectangle

    base_left = min(points_det[:, 0])
    base_top = min(points_det[:, 1])
    base_right = max(points_det[:, 0])
    base_bottom = max(points_det[:, 1])

    # 1b) Fit pose to template
    # Find transformation parameters
    M = points_det.shape[0]
    B = points_det.flatten('F')
    A = np.vstack((np.column_stack((points_reference[:, 0], np.zeros(M), np.ones(M), np.zeros(M))),
                   np.column_stack((np.zeros((M)), points_reference[:, 1], np.zeros(M), np.ones(M)))))

    params = np.linalg.lstsq(A, B, rcond=None)
    params = params[0]
    M = 2
    A2 = np.vstack((np.column_stack((ref_bb[:, 0], np.zeros((M)), np.ones((M)), np.zeros((M)))),
                    np.column_stack((np.zeros((M)), ref_bb[:, 1], np.zeros((M)), np.ones((M))))))

    result = np.matmul(A2, params)

    fit_left = min(result[0:2])
    fit_top = min(result[2:4])
    fit_right = max(result[0:2])
    fit_bottom = max(result[2:4])

    # 2. Fuse bounding boxes
    left = min(base_left, fit_left)
    top = min(base_top, fit_top)
    right = max(base_right, fit_right)
    bottom = max(base_bottom, fit_bottom)

    left = left * 1920
    top = top * 1080
    right = right * 1920
    bottom = bottom * 1080

    height = bottom - top + 1
    width = right - left + 1

    bb = np.array([left, top, width, height])
    return bb


def scale_bb(bb, pose, scaling_factor):
    # Scales bounding box by scaling factor
    newbb = np.zeros(bb.shape)
    newbb[0:2] = bb[0:2] - 0.5 * (scaling_factor - 1) * bb[2:4]
    newbb[2:4] = bb[2:4] * scaling_factor

    # X, Y, strength
    newpose = np.reshape(pose, (18, 3))
    # Scale to original bounding box
    newpose[:, 0] = (newpose[:, 0] - bb[0] / 1920.0) / (bb[2] / 1920.0)
    newpose[:, 1] = (newpose[:, 1] - bb[1] / 1080.0) / (bb[3] / 1080.0)

    # Scale to stretched bounding box
    newpose[:, 0] = (newpose[:, 0] + 0.5 * (scaling_factor - 1)) / scaling_factor
    newpose[:, 1] = (newpose[:, 1] + 0.5 * (scaling_factor - 1)) / scaling_factor
    # Return in the original format
    newpose[newpose[:, 2] == 0, 0:2] = 0
    newpose = np.ravel(newpose)

    return newbb, newpose


def feet_position(boxes):
    x = boxes[0] + 0.5 * boxes[2]
    y = boxes[1] + boxes[3]
    feet = np.array([x, y])
    return feet


def get_bb(img, bb):
    bb = np.round(bb)

    left = np.maximum(0, bb[0]).astype('int')
    right = np.minimum(1920 - 1, bb[0] + bb[2]).astype('int')
    top = np.maximum(0, bb[1]).astype('int')
    bottom = np.minimum(1080 - 1, bb[1] + bb[3]).astype('int')
    if left == right or top == bottom:
        return np.zeros((256, 128, 3))
    snapshot = img[top:bottom, left:right, :]
    return snapshot


def convert_img(img):
    img = img.astype('float')
    img = img / 255.0
    img = img - 0.5
    return img




def load_detections(detections_path):
    with open(detections_path, 'rb') as f:
        detections = pickle.load(f)
    return detections


def num_detections(detections_path):
    with open(detections_path, 'rb') as f:
        detections = pickle.load(f)
    return len(detections)


def detections_generator(video_dir, detections_path):
    reader = CAMPUSVideoReader(video_dir)
    with open(detections_path, 'rb') as f:
        detections = pickle.load(f)
    for i, pose in enumerate(detections):
        img = reader.getFrame(i)
        bb = pose2bb(pose[2:])
        newbb, newpose = scale_bb(bb, pose[2:], 1.25)

        if newbb[2] < 20 or newbb[3] < 20:
            snapshot = np.zeros((256, 128, 3))
        else:
            snapshot = get_bb(img, newbb)
            snapshot = cv2.resize(snapshot, (128, 256))

        yield snapshot



