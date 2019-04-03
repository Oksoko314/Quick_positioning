import h5py
import os
import numpy as np

from L1_tracklets import get_valid_detections, create_tracklets
from duke_utils import load_detections
from config_default import configs
from utils import visiual_tracklets


os.chdir(os.path.join(os.getcwd(), 'triplet_reid'))
dataset_path = os.path.join(configs['dataset_path'], configs['video_name'])
# detections_path = os.path.join(dataset_path, 'detections', 'OpenPose')
detections_path = os.path.join(dataset_path, 'skeletons')
file_dir = os.path.join(dataset_path, configs['file_name'])


def compute_L1_tracklets(features, detections, start_frame, end_frame):
    frame_index = 1
    params = configs['tracklets']

    all_dets = np.copy(detections)

    tracklets = []

    for window_start_frame in range(start_frame, end_frame + 1, params['window_width']):
        window_end_frame = window_start_frame + params['window_width']

        window_inds = np.where(np.logical_and(all_dets[frame_index, :] < window_end_frame, all_dets[frame_index, :]
                                              >= window_start_frame))[0]

        detections_in_window = np.copy(all_dets[:, window_inds]).T

        detections_conf = np.sum(detections_in_window[:, frame_index + 3:np.size(detections_in_window, axis=1)+1:3],
                                 axis=1)
        num_visiable = np.sum(detections_in_window[:, frame_index + 3:np.size(detections_in_window, axis=1)+1:3]
                              > configs['render_threshold'], axis=1)

        vaild = get_valid_detections(detections_in_window, detections_conf, num_visiable, frame_index)
        vaild = np.nonzero(vaild)[0]

        detections_in_window = detections_in_window[vaild, :]

        detections_in_window = np.delete(detections_in_window,
                                         list(range(frame_index + 5, np.size(detections_in_window, 1))), axis=1)
        filtered_detections = detections_in_window

        filtered_features = features[window_inds[vaild], :]

        create_tracklets(configs, filtered_detections, filtered_features,
                         window_start_frame, window_end_frame, frame_index, tracklets)

    return tracklets


if __name__ == '__main__':
    features = h5py.File(file_dir, 'r')['emb']
    detections = load_detections(detections_path)
    tracklets = compute_L1_tracklets(features, detections, 0, configs['total_frame'])
    video = os.path.join(configs['dataset_path'], configs['video_name']+'.mp4')
    visiual_tracklets(tracklets, video)