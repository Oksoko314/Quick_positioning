from Trajectory import *
from config_default import configs
import pickle
import os
import cv2

# must import the class that picked before
from L1_tracklets import SmoothedTracklet


def compute_L2_trajectories(configs, tracklets, start_frame, end_frame):

    trajectories_from_tracklets = tracklets_to_trajectory(tracklets, list(range(1, len(tracklets)+1)))
    trajectories = np.asarray(trajectories_from_tracklets)

    trajectory_config = configs["trajectories"]

    window_start_frame = start_frame - trajectory_config['window_width']
    window_end_frame = start_frame + trajectory_config['window_width']

    while window_start_frame <= end_frame:
        create_trajectories(configs, trajectories, window_start_frame, window_end_frame)
        window_start_frame = window_end_frame - trajectory_config['overlap']
        window_end_frame = window_start_frame + trajectory_config['window_width']
        print('start_frame : {}'.format(window_start_frame))
        print('create_trajectory from {} to {}'.format(window_start_frame, window_end_frame))
    print("=="*10)
    print("create_trajectory complete")
    # Convert trajectories
    tracker_output_raw = trajectories_to_top(trajectories)
    # Interpolate missing detections
    print("begin fill missing detections for trajectories")
    tracker_output_filled = fill_trajectories(tracker_output_raw)
    # Remove spurius tracks
    print("remove trajectories that too short")
    tracker_output_removed = remove_short_tracks(tracker_output_filled,
                                                 trajectory_config['minimum_trajectory_length'])
    # make identities 1-indexed
    _, index = np.unique(tracker_output_removed[:, 1], return_inverse=True)
    tracker_output_removed[:, 0] = index
    tracker_output = tracker_output_removed[np.lexsort((tracker_output_removed[:, 1], tracker_output_removed[:, 0]))]
    print("Compute L2 trajectories complete")
    return tracker_output


def visual_tracker_output(video, tracker_output):
    tracker_output_frames = tracker_output[:, 1]
    cap = cv2.VideoCapture(video)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_tracker_indexs = np.nonzero(tracker_output_frames==i)[0]
        current_trackers = tracker_output[current_tracker_indexs]
        for k, current_tracker in current_trackers:
            id_, _, x, y, w, h = current_tracker
            cv2.rectangle(frame, (x, y), (x+w, y+h), (17*i % 255, 31*i % 255, 53*i % 255))
        cv2.imshow('result', frame)
        cv2.waitKey(10)
        i += 1


if __name__ == '__main__':
    print(os.getcwd())
    if os.path.getsize("tracklets") > 0:
        with open("tracklets", 'rb') as dbfile:
            tracklets = pickle.load(dbfile)
        start_frame = 122178
        end_frame = 181998
        compute_L2_trajectories(configs, tracklets, start_frame, end_frame)


