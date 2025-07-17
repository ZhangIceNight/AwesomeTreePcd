
from models import KPFCNN
from types import SimpleNamespace

from os.path import join



import numpy as np
import torch
from torch.utils.data import Dataset

class RandomPointCloudDataset(Dataset):
    def __init__(self, num_samples=100, num_points=4096, config=None, classification=False):
        self.num_samples = num_samples
        self.num_points = num_points
        self.config = config
        self.classification = classification

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成随机点云，范围[0,1)，numpy格式
        point_set = np.random.rand(self.num_points, 3).astype(np.float32)

        # 生成特征，根据config.in_features_dim
        if self.config is None:
            in_features_dim = 1
        else:
            in_features_dim = getattr(self.config, 'in_features_dim', 1)

        if in_features_dim == 1:
            features = np.ones((self.num_points, 1), dtype=np.float32)
        elif in_features_dim == 4:
            features = np.concatenate([np.ones((self.num_points, 1), dtype=np.float32), point_set], axis=1)
        elif in_features_dim == 7:
            # 这里没有法算法线，先用随机数代替
            normals = np.random.rand(self.num_points, 3).astype(np.float32)
            features = np.concatenate([np.ones((self.num_points, 1), dtype=np.float32), point_set, normals], axis=1)
        else:
            # 其他维度简单生成全1特征
            features = np.ones((self.num_points, in_features_dim), dtype=np.float32)

        if self.classification:
            # 随机生成0~9分类标签整数（你根据类别数量改）
            cls = np.random.randint(0, 10, size=1).astype(np.int64)
            return point_set, features, cls[0]
        else:
            # 分割标签，点级别，假设2类（0和1）随机分配
            seg = np.random.randint(0, 2, size=self.num_points).astype(np.int32)
            return point_set, features, seg


class Config:
    """
    Class containing the parameters you want to modify for this dataset
    """

    ##################
    # Input parameters
    ##################
    # Dataset name
    dataset = ''

    # Type of network model
    network_model = ''

    # Number of classes in the dataset
    num_classes = 3

    # Dimension of input points
    in_points_dim = 3

    # Dimension of input features
    in_features_dim = 3

    # Radius of the input sphere (ignored for models, only used for point clouds)
    in_radius = 1.0

    # Number of CPU threads for the input pipeline
    input_threads = 8

    ##################
    # Model parameters
    ##################

    # Architecture definition. List of blocks
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb_deformable',
                    'resnetb_deformable_strided',
                    'resnetb_deformable',
                    'resnetb_deformable_strided',
                    'resnetb_deformable',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']

    # Dimension of the first feature maps
    first_features_dim = 64

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.99

    # For segmentation models : ratio between the segmented area and the input area
    segmentation_ratio = 1.0

    ###################
    # KPConv parameters
    ###################

    # First size of grid used for subsampling
    first_subsampling_dl = 0.02

    # Radius of the kernels in the first layer (deprecated)
    first_kernel_radius = 0.1

    # Number of points in the kernels
    num_kernel_points = 15

    # density of neighbors in kernel range
    # For each layer, support points are subsampled on a grid with dl = kernel_radius / density_parameter
    density_parameter = 3.0

    # Kernel point influence radius
    KP_extent = 1.0

    # Influence function when d < KP_extent. ('constant', 'linear', 'gaussian') When d > KP_extent, always zero
    KP_influence = 'gaussian'

    # Behavior of convolutions in ('closest', 'sum')
    # Decide if you sum all kernel point influences, or if you only take the influence of the closest KP
    convolution_mode = 'closest'

    # Fixed points in the kernel : 'none', 'center' or 'verticals'
    fixed_kernel_points = 'center'

    # Can the network learn kernel dispositions (deprecated)
    trainable_positions = False

    # Use modulateion in deformable convolutions
    modulated = False

    #####################
    # Training parameters
    #####################

    # Network optimizer parameters (learning rate and momentum)
    learning_rate = 1e-4
    momentum = 0.9

    # Learning rate decays. Dictionary of all decay values with their epoch {epoch: decay}.
    lr_decays = {200: 0.2, 300: 0.2}

    # Gradient clipping value (negative means no clipping)
    grad_clip_norm = 100.0

    # Augmentation parameters
    augment_scale_anisotropic = True
    augment_scale_min = 0.9
    augment_scale_max = 1.1
    augment_symmetries = [False, False, False]
    augment_rotation = 'vertical'
    augment_noise = 0.005
    augment_occlusion = 'planar'
    augment_occlusion_ratio = 0.2
    augment_occlusion_num = 1
    augment_color = 0.7
    augment_shift_range = 0

    # Regularization loss importance
    weights_decay = 1e-6

    # Gaussian loss
    gaussian_decay = 1e-3

    # Type of output loss with regard to batches when segmentation
    batch_averaged_loss = False

    # Point loss DPRECATED
    points_loss = ''
    points_decay = 1e-2

    # Offset regularization loss
    offsets_loss = 'permissive'
    offsets_decay = 1e-2

    # Number of batch
    batch_num = 10

    # Maximal number of epochs
    max_epoch = 1000

    # Number of steps per epochs
    epoch_steps = 1000

    # Number of validation examples per epoch
    validation_size = 100

    # Number of epoch between each snapshot
    snapshot_gap = 50

    # Do we nee to save convergence
    saving = True
    saving_path = None

    # wj20250714
    conv_radius = 3.0
    aggregation_mode = 'sum'
    class_w = []
    deform_fitting_mode = 'point2point'
    deform_fitting_power = '1.0'
    deform_lr_factor = '0.5'
    repulse_extent = '1.0'

    def __init__(self):
        """
        Class Initialyser
        """

        # Number of layers
        self.num_layers = len([block for block in self.architecture if 'pool' in block or 'strided' in block]) + 1

    # def load(self, path):

    #     filename = join(path, 'parameters.txt')
    #     with open(filename, 'r') as f:
    #         lines = f.readlines()

    #     # Class variable dictionary
    #     for line in lines:
    #         line_info = line.split()
    #         if len(line_info) > 1 and line_info[0] != '#':

    #             if line_info[2] == 'None':
    #                 setattr(self, line_info[0], None)

    #             elif line_info[0] == 'lr_decay_epochs':
    #                 self.lr_decays = {int(b.split(':')[0]): float(b.split(':')[1]) for b in line_info[2:]}

    #             elif line_info[0] == 'architecture':
    #                 self.architecture = [b for b in line_info[2:]]

    #             elif line_info[0] == 'augment_symmetries':
    #                 self.augment_symmetries = [bool(int(b)) for b in line_info[2:]]

    #             elif line_info[0] == 'num_classes':
    #                 if len(line_info) > 3:
    #                     self.num_classes = [int(c) for c in line_info[2:]]
    #                 else:
    #                     self.num_classes = int(line_info[2])

    #             else:

    #                 attr_type = type(getattr(self, line_info[0]))
    #                 if attr_type == bool:
    #                     setattr(self, line_info[0], attr_type(int(line_info[2])))
    #                 else:
    #                     setattr(self, line_info[0], attr_type(line_info[2]))

    #     self.saving = True
    #     self.saving_path = path
    #     self.__init__()

    # def save(self, path):

    #     with open(join(path, 'parameters.txt'), "w") as text_file:

    #         text_file.write('# -----------------------------------#\n')
    #         text_file.write('# Parameters of the training session #\n')
    #         text_file.write('# -----------------------------------#\n\n')

    #         # Input parameters
    #         text_file.write('# Input parameters\n')
    #         text_file.write('# ****************\n\n')
    #         text_file.write('dataset = {:s}\n'.format(self.dataset))
    #         text_file.write('network_model = {:s}\n'.format(self.network_model))
    #         if type(self.num_classes) is list:
    #             text_file.write('num_classes =')
    #             for n in self.num_classes:
    #                 text_file.write(' {:d}'.format(n))
    #             text_file.write('\n')
    #         else:
    #             text_file.write('num_classes = {:d}\n'.format(self.num_classes))
    #         text_file.write('in_points_dim = {:d}\n'.format(self.in_points_dim))
    #         text_file.write('in_features_dim = {:d}\n'.format(self.in_features_dim))
    #         text_file.write('in_radius = {:.3f}\n'.format(self.in_radius))
    #         text_file.write('input_threads = {:d}\n\n'.format(self.input_threads))

    #         # Model parameters
    #         text_file.write('# Model parameters\n')
    #         text_file.write('# ****************\n\n')

    #         text_file.write('architecture =')
    #         for a in self.architecture:
    #             text_file.write(' {:s}'.format(a))
    #         text_file.write('\n')
    #         text_file.write('num_layers = {:d}\n'.format(self.num_layers))
    #         text_file.write('first_features_dim = {:d}\n'.format(self.first_features_dim))
    #         text_file.write('use_batch_norm = {:d}\n'.format(int(self.use_batch_norm)))
    #         text_file.write('batch_norm_momentum = {:.3f}\n\n'.format(self.batch_norm_momentum))
    #         text_file.write('segmentation_ratio = {:.3f}\n\n'.format(self.segmentation_ratio))

    #         # KPConv parameters
    #         text_file.write('# KPConv parameters\n')
    #         text_file.write('# *****************\n\n')

    #         text_file.write('first_subsampling_dl = {:.3f}\n'.format(self.first_subsampling_dl))
    #         text_file.write('num_kernel_points = {:d}\n'.format(self.num_kernel_points))
    #         text_file.write('density_parameter = {:.3f}\n'.format(self.density_parameter))
    #         text_file.write('fixed_kernel_points = {:s}\n'.format(self.fixed_kernel_points))
    #         text_file.write('KP_extent = {:.3f}\n'.format(self.KP_extent))
    #         text_file.write('KP_influence = {:s}\n'.format(self.KP_influence))
    #         text_file.write('convolution_mode = {:s}\n'.format(self.convolution_mode))
    #         text_file.write('trainable_positions = {:d}\n\n'.format(int(self.trainable_positions)))
    #         text_file.write('modulated = {:d}\n\n'.format(int(self.modulated)))

    #         # Training parameters
    #         text_file.write('# Training parameters\n')
    #         text_file.write('# *******************\n\n')

    #         text_file.write('learning_rate = {:f}\n'.format(self.learning_rate))
    #         text_file.write('momentum = {:f}\n'.format(self.momentum))
    #         text_file.write('lr_decay_epochs =')
    #         for e, d in self.lr_decays.items():
    #             text_file.write(' {:d}:{:f}'.format(e, d))
    #         text_file.write('\n')
    #         text_file.write('grad_clip_norm = {:f}\n\n'.format(self.grad_clip_norm))

    #         text_file.write('augment_symmetries =')
    #         for a in self.augment_symmetries:
    #             text_file.write(' {:d}'.format(int(a)))
    #         text_file.write('\n')
    #         text_file.write('augment_rotation = {:d}\n'.format(self.augment_rotation))
    #         text_file.write('augment_noise = {:f}\n'.format(self.augment_noise))
    #         text_file.write('augment_occlusion = {:s}\n'.format(self.augment_occlusion))
    #         text_file.write('augment_occlusion_ratio = {:.3f}\n'.format(self.augment_occlusion_ratio))
    #         text_file.write('augment_occlusion_num = {:d}\n'.format(self.augment_occlusion_num))
    #         text_file.write('augment_scale_anisotropic = {:d}\n'.format(int(self.augment_scale_anisotropic)))
    #         text_file.write('augment_scale_min = {:.3f}\n'.format(self.augment_scale_min))
    #         text_file.write('augment_scale_max = {:.3f}\n'.format(self.augment_scale_max))
    #         text_file.write('augment_color = {:.3f}\n\n'.format(self.augment_color))

    #         text_file.write('weights_decay = {:f}\n'.format(self.weights_decay))
    #         text_file.write('gaussian_decay = {:f}\n'.format(self.gaussian_decay))
    #         text_file.write('batch_averaged_loss = {:d}\n'.format(int(self.batch_averaged_loss)))
    #         text_file.write('offsets_loss = {:s}\n'.format(self.offsets_loss))
    #         text_file.write('offsets_decay = {:f}\n'.format(self.offsets_decay))
    #         text_file.write('batch_num = {:d}\n'.format(self.batch_num))
    #         text_file.write('max_epoch = {:d}\n'.format(self.max_epoch))
    #         if self.epoch_steps is None:
    #             text_file.write('epoch_steps = None\n')
    #         else:
    #             text_file.write('epoch_steps = {:d}\n'.format(self.epoch_steps))
    #         text_file.write('validation_size = {:d}\n'.format(self.validation_size))
    #         text_file.write('snapshot_gap = {:d}\n'.format(self.snapshot_gap))


config = Config()

model = KPFCNN(config=config, lbl_values=[0,1,2], ign_lbls=[0])


points_tensor = torch.rand(B, N, 3, dtype=torch.float32)
x = {
    'points': points_tensor,
    'features': points_tensor
}

print(model)

dset = RandomPointCloudDataset(num_samples=100, num_points=4096, config=config, classification=True)
dataloader = get_dataloader(dset, batch_size=4)