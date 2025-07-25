#
#
#      0=================================0
#      |            Sen-net              |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Define network architectures
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Sen-net's architecture is based on Hugues THOMAS's KPFCNN version in 06/03/2020
#

from models.Sennet_blocks import *
import numpy as np


def p2p_fitting_regularizer(net):
    return net.deform_fitting_power * (2 * fitting_loss + repulsive_loss)



class KPFCNN(nn.Module):
    """
    Class defining KPFCNN
    """

    def __init__(self, config, lbl_values, ign_lbls):
        super(KPFCNN, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls) #  the length of all class minus the length of ignored class  to get the number of class culculated in Loss function -wj20240714 

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []
        self.encoder_skip_dims_2 = []
        self.encoder_skips_2 = []
        self.encoder_skip_r = []
        self.encoder_skip_layer = []
        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)
                self.encoder_skip_dims_2.append(in_dim)
                self.encoder_skip_r.append(r)
                self.encoder_skip_layer.append(layer)
            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2
        #####################
        # semantic-driven detail enrichment module
        #####################
        self.num = 4
        self.maxpool_blocks = nn.ModuleList()
        self.feats_att = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        self.feats_att_down = nn.ModuleList()
        self.d_mlp_blocks = nn.ModuleList()
        for i in range(self.num):
            self.maxpool_blocks.append(block_decider('max_pool_2',
                                                     self.encoder_skip_r[i],
                                                     self.encoder_skip_dims_2[i],
                                                     self.encoder_skip_dims_2[i + 1],
                                                     self.encoder_skip_layer[i],
                                                     config))
            #print(self.encoder_skip_layer[i])

            self.feats_att.append(block_decider('unary',
                                                self.encoder_skip_r[i + 1],
                                                self.encoder_skip_dims_2[i + 1],
                                                1,
                                                self.encoder_skip_layer[i + 1],
                                                config))
            self.feats_att_down.append(block_decider('unary',
                                                     self.encoder_skip_r[i],
                                                     self.encoder_skip_dims_2[i],
                                                     1,
                                                     self.encoder_skip_layer[i],
                                                     config))
        #print(self.maxpool_blocks[0])

        for i in range(self.num):
            self.upsample_blocks.append(block_decider('nearest_upsample',
                                                      self.encoder_skip_r[i],
                                                      self.encoder_skip_dims_2[i],
                                                      self.encoder_skip_dims[i + 1],
                                                      self.encoder_skip_layer[i + 1],
                                                      config))

        for i in range(self.num - 1):
            self.d_mlp_blocks.append(block_decider('unary',
                                                      self.encoder_skip_r[i],
                                                      self.encoder_skip_dims_2[i + 1],
                                                      self.encoder_skip_dims_2[i],
                                                      self.encoder_skip_layer[i],
                                                      config))
            # print(self.encoder_skip_dims_2[i+1])
            # print(self.encoder_skip_dims_2[i])
        i = 3
        self.d_mlp_blocks.append(block_decider('unary',
                                                          self.encoder_skip_r[i],
                                                          2*(self.encoder_skip_dims_2[i + 1]),
                                                          self.encoder_skip_dims_2[i],
                                                          self.encoder_skip_layer[i],
                                                          config))

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += 2*(self.encoder_skip_dims[layer])
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_softmax = UnaryBlock(config.first_features_dim, self.C, False, 0, no_relu=True)

        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

        # Choose segmentation loss
        if len(config.class_w) > 0:
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def SDFM(self, master_feats, large_feats, cur_ind):

        _feats = self.feats_att[cur_ind](master_feats)
        _large_feats = self.feats_att_down[cur_ind](large_feats)
        att_map_1 = torch.sigmoid(_feats * _large_feats)
        final_feats1 = large_feats * att_map_1[:, None]
        feats = torch.cat([master_feats, final_feats1], dim=1)
        att_map_2 = torch.sigmoid(_feats + _large_feats)
        final_feats2 = large_feats * att_map_2[:, None]
        feats = torch.cat([feats, final_feats2], dim=1)


        return feats

    def SDFM2(self, master_feats, large_feats, cur_ind):

        _feats = self.feats_att[cur_ind](master_feats)
        _large_feats = self.feats_att_down[cur_ind](large_feats)
        att_map_1 = torch.sigmoid(_feats * _large_feats)
        final_feats1 = large_feats * att_map_1[:, None]
        feats = torch.cat([master_feats, final_feats1], dim=1)
        feats = torch.cat([feats, large_feats], dim=1)


        return feats

    def SDFM3(self, master_feats, large_feats, cur_ind):

        _feats = self.feats_att[cur_ind](master_feats)
        _large_feats = self.feats_att_down[cur_ind](large_feats)
        att_map_2 = torch.sigmoid(_feats + _large_feats)
        final_feats2 = large_feats * att_map_2[:, None]
        feats = torch.cat([master_feats, final_feats2], dim=1)
        feats = torch.cat([feats, large_feats], dim=1)


        return feats

    def forward(self, batch):

        # Get input features
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        skip_x = []
        skip_x2 = []
        '''
        encoder
        '''
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
                skip_x2.append(x)
            x = block_op(x, batch)
        skip_x2.append(x)

        '''
        semantic block
        '''
        for i in range(self.num):
            pool_op = self.maxpool_blocks[i]
            feat = self.SDFM(skip_x2[i + 1], pool_op(skip_x2[i], batch), i)

        skip_feat = []

        for j in range(self.num - 1, -1, -1):

            up_op = self.upsample_blocks[j]
            feat = up_op(feat, batch)

            d_op = self.d_mlp_blocks[j]
            feat = d_op(feat, batch)

            skip_feat.append(feat)


        '''
        decoder block
        '''
        temp = 0
        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
                x = torch.cat([x, skip_feat[temp]], dim=1)
                temp = temp + 1
            x = block_op(x, batch)

        # Head of network
        x = self.head_mlp(x)
        x = self.head_softmax(x)

        return x

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.unsqueeze(0)

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    def accuracy(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total
    
if __name__ == '__main__':
    model = KPFCNN(config=configDict, lbl_values=None, ign_lbls=None)
    print(model)