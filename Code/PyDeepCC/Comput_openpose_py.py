import sys
import cv2
import copy
import os
import pickle
import numpy as np

from config_default import configs

sys.path.append('/home/fyq/openpose/build/python/')
from openpose import pyopenpose as op

params = dict()
params["model_folder"] = "/home/fyq/openpose/models"

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


def skeleton_from_video(video_path, icam = 0, visual=True):
    cap = cv2.VideoCapture(video_path)

    all_poseKeypoints = []

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        datum = op.Datum()
        imageToProcess = copy.copy(frame)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])

        if visual:
            cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", datum.cvOutputData)
            key = cv2.waitKey(15)
            if key == 27:
                break

        poseKeyPoints = datum.poseKeypoints

        if isinstance(poseKeyPoints, np.ndarray):
            for k, pose in enumerate(poseKeyPoints):
                pose = np.hstack((icam, i, pose))
                all_poseKeypoints.append(pose)
        i += 1
    return np.asarray(all_poseKeypoints)


def compute_openpose(skeleton_file, video_file):
    skeletons = skeleton_from_video(video_file, visual=False)

    with open(skeleton_file, 'wb') as f:
        pickle.dump(f, skeletons)


if __name__ == '__main__':
    os.chdir(os.path.join(os.getcwd(), 'triplet_reid'))

    video_dir = os.path.join(configs['dataset_path'], configs['video_name']+'.mp4')
    dataset_path = os.path.join(configs['dataset_path'], configs['video_name'])

    skeletons = skeleton_from_video(video_dir)

    with open(os.path.join('skeletons'), 'wb') as f:
        pickle.dump(f, skeletons)
