from Trajectory import *
from config_default import configs
import pickle
import os
import cv2
from L1_tracklets import SmoothedTracklet


def compute_L2_trajectories(configs, tracklets, start_frame, end_frame):
    trajectories_from_tracklets = tracklets_to_trajectory(tracklets, list(range(1, len(tracklets) + 1)))
    trajectories = trajectories_from_tracklets

    while start_frame <= end_frame:
        trajectories = create_trajectories(configs, trajectories, start_frame, end_frame)
        start_frame = end_frame - configs['trajectories']['overlap']
        end_frame = start_frame + configs['trajectories']['window_width']

    # Convert trajectories
    tracker_output_raw = trajectories_to_top(trajectories)
    # Interpolate missing detections
    tracker_output_filled = fill_trajectories(tracker_output_raw)
    # Remove spurius tracks
    tracker_output_removed = remove_short_tracks(tracker_output_filled,
                                                 configs['trajectories']['minimum_trajectory_length'])

    _, index = np.unique(tracker_output_removed[:, 0], return_inverse=True)
    tracker_output_removed[:, 0] = index
    tracker_output = tracker_output_removed[tracker_output_removed[:, [0, 1]].argsort(),]
    return tracker_output


def visual_trajectories(tracker_final, video_dir, start_frame, end_frame):
    cap = cv2.VideoCapture(video_dir)
    tracker_frame_inds = tracker_final[:, 1]
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if start_frame <= i < end_frame:
            current_trackers = tracker_final[np.nonzero(tracker_frame_inds == i)[0]]

            for k, current_tracker in enumerate(current_trackers):
                id_, _, x, y, w, h = current_tracker
                cv2.rectangle(frame, (x, y), (x + w, y + h), (17 * id_ % 255, 31 * id_ % 255, 53 * id_ % 255))

            cv2.imshow('DeepCC final result', frame)
            key = cv2.waitKey(15)
            if key == 27:
                break
        i += 1


if __name__ == '__main__':

    os.chdir(os.path.join(os.getcwd(), 'triplet_reid'))

    dataset_path = os.path.join(configs['dataset_path'], configs['video_name'])

    if os.path.getsize(os.path.join(dataset_path, configs['tracklets_out'])) > 0:
        with open(os.path.join(dataset_path, configs['tracklets_out']), 'rb') as dbfile:
            tracklets = pickle.load(dbfile)

        start_frame = 0
        end_frame = configs['total_frame']
        tracker_final = compute_L2_trajectories(configs, tracklets, start_frame, end_frame)

        tracker_output_file = os.path.join(dataset_path, configs['trajectory_out'])

        with open(tracker_output_file, 'wb') as f:
            pickle.dump(tracker_final, f)

        video_dir = os.path.join(configs['dataset_path'], configs['video_name'] + '.mp4')

        visual_trajectories(tracker_final, video_dir, start_frame, end_frame)
