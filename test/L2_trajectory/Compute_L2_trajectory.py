from Trajectory import *
from config_default import configs
import pickle
import os
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


if __name__ == '__main__':
    print(os.getcwd())
    if os.path.getsize("tracklets") > 0:
        with open("tracklets", 'rb') as dbfile:
            tracklets = pickle.load(dbfile)
        start_frame = 122178
        end_frame = 181998
        compute_L2_trajectories(configs, tracklets, start_frame, end_frame)


