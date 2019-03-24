import numpy as np
from sklearn.cluster import k_means
from scipy.spatial.distance import cdist
from collections import Counter
import time

from utils import get_appearance_matrix, get_space_time_affinity
from L1_tracklets.KernighanLin import KernighanLin


def solve_in_groups(configs, tracklets, labels):
    params = configs["trajectories"]
    if len(tracklets) < params["appearance_groups"]:
        configs['appearance_groups'] = 1

    feature_vectors = np.array([tracklet.feature for tracklet in tracklets])
    feature_vectors = np.reshape(feature_vectors, (len(tracklets), 128))

    # adaptive number of appearance groups
    if params["appearance_groups"] == 0:
        # Increase number of groups until no group is too large to solve
        while True:
            params["appearance_groups"] += 1
            apperance_feature, appearance_groups,_ = k_means(feature_vectors, n_clusters=params["appearance_groups"],
                                                             n_jobs=-1)
            uid, freq = list(zip(*Counter(appearance_groups).items()))
            largest_group_size = max(freq)
            # The BIP solver might run out of memory for large graphs
            if largest_group_size <= 150:
                break
    else:
        # fixed number of appearance groups
        apperance_feature, appearance_groups, _ = k_means(feature_vectors, n_clusters=params["appearance_groups"],
                                                          n_jobs=-1)
    # solve separately for each appearance group
    all_groups = np.unique(appearance_groups)

    result_appearance = []
    for i, group in enumerate(all_groups):
        print("merging tracklets in appearance group {}\n".format(i))
        indices = np.nonzero(appearance_groups == group)
        same_labels = cdist(labels[indices], labels[indices]) == 0

        # compute appearance and spacetime scores
        appearance_affinity = get_appearance_matrix(feature_vectors[indices], params["threshold"])
        spacetime_affinity, impossibility_matrix, indifference_matrix = get_space_time_affinity(tracklets[indices],
                                                                                                params["beta"],
                                                                                                params["speed_limit"],
                                                                                                params["indifference_time"])
        # compute the correlation matrix
        correlation_matrix = appearance_affinity + spacetime_affinity - 1
        correlation_matrix = correlation_matrix * indifference_matrix

        correlation_matrix[impossibility_matrix == 1] = -np.inf
        correlation_matrix[same_labels] = 1

        # just use KL optimizer now
        labels = KernighanLin(correlation_matrix)
        result_appearance.append({"labels": labels, "observations": indices})

        result = {"labels": [], "observations": []}

        for i in range(np.size(np.unique(appearance_groups))):
            merge_results(result, result_appearance[i])

        # ToDO edit
        id_ = sorted(result["observations"])
        result["observations"] = result["observations"][id_]
        result["labels"] = result["labels"][id_]
        return result


def merge_results(result1, result2):
    maxinum_label = np.max(result1.labels)
    if np.size(maxinum_label) == 0:
        maxinum_label = 0

    result1["labels"].append(maxinum_label + result2["labels"])
    result1["observations"].append(result2["observations"])


def find_trajectories_in_window(input_trajectories, start_time, end_time):
    trajecotry_start_frame = np.array([input_trajectory.start_frame for input_trajectory in input_trajectories],
                                      dtype=np.int)
    trajecotry_end_frame = np.array([input_trajectory.end_frame for input_trajectory in input_trajectories],
                                      dtype=np.int)
    trajectories_ind = np.where(np.logical_and(trajecotry_end_frame >= start_time, trajecotry_start_frame <= end_time))
    return trajectories_ind


class Trajectory:
    def __init__(self, start_frame, end_frame, segment_start, segment_end):
        self.tracklets = []
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.segment_start = segment_start
        self.segment_end = segment_end
        self.feature = None


def tracklets_to_trajectory(tracklets, labels):
    unique_labels = np.unique(labels)
    trajectories = []
    for i, label in enumerate(unique_labels):
        trajectory = Trajectory(np.inf, -np.inf, np.inf, -np.inf)
        tracklet_indices = np.nonzero(labels == label)
        for j, ind in enumerate(tracklet_indices):
            trajectory.tracklets.append(tracklets[ind])
            trajectory.start_frame = min(trajectory.start_frame, tracklets[ind].start)
            trajectory.end_frame = max(trajectory.end_frame, tracklets[ind].finish)
            trajectory.segment_start = min(trajectory.segment_start, tracklets[ind].segment_start)
            trajectory.feature = tracklets[ind].feature
        trajectories.append(trajectory)
    return trajectories


def create_trajectories(configs, input_trajectories, start_frame, end_frame):
    current_trajectories_ind = find_trajectories_in_window(input_trajectories, start_frame, end_frame)
    current_trajectories = input_trajectories[current_trajectories_ind]
    if len(current_trajectories) <=1:
        out_trajectories = input_trajectories
        return out_trajectories

    # select tracklets that will be selected in association.For previously
    # computed trajectories we select only the last three tracklets.
    is_assocations = []
    tracklets = []
    tracklet_labels = []
    for i, current_trajectory in enumerate(current_trajectories):
        for j, tracklet in enumerate(current_trajectory.tracklets):
            tracklets.append(tracklet)
            tracklet_labels.append(i)

            if j >= len(current_trajectory.tracklets)-4:
                is_assocations.append(True)
            else:
                is_assocations.append(False)

    # solve the graph partitioning problem for each appearance group
    result = solve_in_groups(configs,
                             [tracklet for tracklet, is_assocation in zip(tracklets, is_assocations) if is_assocation],
                             [tracklet_label for tracklet_label, is_assocation in zip(tracklet_labels, is_assocations)
                              if is_assocation])

    # merge back solution. Tracklets that were associated are now merged back
    # with the rest of tracklets that were sharing the same trajectory
    labels = tracklet_labels
    # labels[is_assocations] = result["labels"]

    count = 0
    for i, is_assocation in enumerate(is_assocations):
        if is_assocation:
            labels[tracklet_labels == tracklet_labels[i]] = result["labels"][count]
            count += 1
    # merge co-identified tracklets to extended tracklets
    new_trajectories = tracklets_to_trajectory(tracklets, labels)
    smooth_trajectories = recompute_trajectories(new_trajectories)

    output_trajectories = input_trajectories
    np.delete(output_trajectories, current_trajectories_ind)
    output_trajectories = np.vstack(output_trajectories, smooth_trajectories)

    # show merged tracklets in window
    # TODO visualize