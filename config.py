# config.py

cfg = {
    'experiment_name': 'pls_pointnet_cls',
    'device': 'cuda',  # or 'cpu'
    'num_points': 1024,
    'batch_size': 16,
    'num_epochs': 200,
    'learning_rate': 0.001,
    'optimizer': 'Adam',
    'weight_decay': 1e-4,
    'use_normals': False,
    'use_uniform_sample': True,
    'data_root': 'data/PLSDataset',
    'log_dir': 'log',
    'model_name': 'pointnet',
    'num_workers': 4,
    'save_path': 'log/checkpoints/best_model.pth'
}

