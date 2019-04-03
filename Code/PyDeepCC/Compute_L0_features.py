from config_default import configs
import os
import glob
import scipy.io as sio
import h5py
import numpy as np
from functools import partial
from triplet_reid.embed_detections import embed_detections
from triplet_reid.duke_utils import detections_generator, num_detections
from Comput_openpose_py import compute_openpose


os.chdir(os.path.join(os.getcwd(), 'triplet_reid'))

print(os.getcwd())

if __name__ == '__main__':
    dataset_path = os.path.join(configs['dataset_path'], configs['video_name'])

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)

    detections_file = os.path.join(dataset_path, 'skeletons')
    # print(detections_path)
    video_dir = os.path.join(dataset_path, configs['video_name']+'mp4')
    if not os.path.exists(detections_file):
        compute_openpose(detections_file, video_dir)

    render_threshold = configs['render_threshold']
    width = configs['video_width']
    height = configs['video_height']

    net_config = configs['net']
    experiment_root = net_config['experiment_root']
    file_name = os.path.join(dataset_path, configs['file_name'])

    num_detections = num_detections(detections_file)
    detection_generator = partial(detections_generator, video_dir, detections_file)
    embed_detections(experiment_root, detection_generator, num_detections, file_name)
