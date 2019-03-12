configs = {
    'net': {
        "train_set": 'data/duke_train.csv',
        "image_root": '/home/fyq/Documents/Datasets/DukeMTMC/DukeMTMC-reID',
        "model_name": 'resnet_v1_50',
        'initial_checkpoint': 'resnet_v1_50.ckpt',
        'experiment_root': 'experiments/demo_weighted_triplet',
        'embedding_dim': 128,
        'batch_p': 18,
        'batch_k': 4,
        'pre_crop_height': 288,
        'pre_crop_width': 144,
        'input_width': 128,
        'input_height': 256,
        'margin': 'soft',
        'metric': 'euclidean',
        'loss': 'weighted_triplet',
        'learning_rate': 0.0003,
        'train_iterations': 25000,
        'decay_start_iteration': 15000,
        'gpu_device': 0,
        'augment': True,
        'resume': False,
        'checkpoint_frequency': 1000,
        'hard_pool_size': 0,
    },

    'tracklets': {
        'window_width': 50,
        'min_length': 5,
        'alpha': 1,
        'beta': 2,
        'cluster_coeff': 0.75,
        'nearest_neighbors': 8,
        'speed_limit': 20,
        'threshold': 8
    },
    'dataset_path': '/home/fyq/Documents/Datasets/DukeMTMC/',
    'file_name': '../experiments/demo/L0-features/features{}.h5',
    'render_threshold': 0.005,
    'video_width': 1920,
    'video_height': 1080,
}