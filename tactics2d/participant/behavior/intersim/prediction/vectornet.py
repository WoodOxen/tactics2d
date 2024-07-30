from typing import Dict, List, Tuple, NamedTuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from prediction.decoder import Decoder, DecoderResCat
from prediction.lib import MLP, GlobalGraph, LayerNorm, SubGraph, CrossAttention, GlobalGraphRes
import prediction.utils as utils

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=1, affine=True),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=1, affine=True),
            nn.ReLU()
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=1, affine=True),
            nn.ReLU()
        )  # 14 x 14
        self.double_conv2 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=1, affine=True),
            nn.ReLU()
        )  # 28 x 28
        self.double_conv3 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=1, affine=True),
            nn.ReLU()
        )  # 56 x 56
        self.double_conv4 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()
        )  # 112 x 112

        self.double_conv5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=1, affine=True),
            nn.ReLU(),
        )  # 224 x 224

        # self.raster_scale = args.other_params['raster_scale']
        # assert isinstance(self.raster_scale, int)

    def forward(self, x, concat_features):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.upsample(out)  # block 1
        out = torch.cat((out, concat_features[-1]), dim=1)
        out = self.double_conv1(out)
        out = self.upsample(out)  # block 2
        out = torch.cat((out, concat_features[-2]), dim=1)
        out = self.double_conv2(out)
        out = self.upsample(out)  # block 3
        out = torch.cat((out, concat_features[-3]), dim=1)
        out = self.double_conv3(out)
        out = self.upsample(out)  # block 4
        out = torch.cat((out, concat_features[-4]), dim=1)
        out = self.double_conv4(out)
        # return out
        out = self.upsample(out)  # block 5
        out = torch.cat((out, concat_features[-5]), dim=1)
        out = self.double_conv5(out)
        return out


class CNNDownSampling(nn.Module):
    def __init__(self):
        super(CNNDownSampling, self).__init__()
        import torchvision.models as models
        self.cnn = models.vgg16(pretrained=False, num_classes=128)
        self.cnn.features = self.cnn.features[1:]
        in_channels = 60 + 90
        # if 'train_reactor' in args.other_params and 'raster_inf' in args.other_params:
        #     in_channels = 60 + 90
        # elif 'train_direct_reactor' in args.other_params:
        #     in_channels = 60 + 90
        # elif 'detect_all_inf' in args.other_params:
        #     in_channels = 60 + 90
        # else:
        #     in_channels = 60
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        output = self.cnn(x)
        assert output.shape == (len(x), 128), output.shape
        return output


class CNNEncoder(nn.Module):
    """docstring for ClassName"""

    def __init__(self):
        super(CNNEncoder, self).__init__()
        import torchvision.models as models
        features = list(models.vgg16_bn(pretrained=False).features)
        in_channels = 60 + 90
        # if 'train_reactor' in args.other_params and 'raster_inf' in args.other_params:
        #     in_channels = 60 + 90
        # elif 'train_direct_reactor' in args.other_params:
        #     in_channels = 60 + 90
        # elif 'detect_all_inf' in args.other_params:
        #     in_channels = 60 + 90
        # else:
        #     in_channels = 60
        # if args.nuscenes:
        #     in_channels = 3
        # if 'raster-in_c' in args.other_params:
        #     in_channels = args.other_params['raster-in_c']
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        )
        self.features = nn.ModuleList(features)[1:]  # .eval()
        # print (nn.Sequential(*list(models.vgg16_bn(pretrained=True).children())[0]))
        # self.features = nn.ModuleList(features).eval()
        self.decoder = RelationNetwork()

    def forward(self, x):
        results = []
        x = self.layer1(x)
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {4, 11, 21, 31, 41}:
                results.append(x)

        output = self.decoder(x, results)
        output = output.permute(0, 2, 3, 1)
        assert output.shape == (len(x), 224, 224, 128), output.shape
        return output


class VectorNet(nn.Module):
    r"""
    VectorNet

    It has two main components, sub graph and global graph.

    Sub graph encodes a polyline as a single vector.
    """

    def __init__(self, args_: utils.Args):
        super(VectorNet, self).__init__()
        global args
        args = args_
        hidden_size = 128

        self.sub_graph = SubGraph(args, hidden_size)
        self.global_graph = GlobalGraph(hidden_size)
        self.global_graph = GlobalGraphRes(hidden_size)

        self.laneGCN_A2L = CrossAttention(hidden_size)
        self.laneGCN_L2L = GlobalGraphRes(hidden_size)
        self.laneGCN_L2A = CrossAttention(hidden_size)

        self.cnn_encoder = CNNEncoder()

        self.decoder = Decoder(args, self)

    def forward_encode_sub_graph(self, mapping: List[Dict], matrix: List[np.ndarray], polyline_spans: List[slice],
                                 device, batch_size) -> Tuple[List[Tensor], List[Tensor]]:
        """
        :param matrix: each value in list is vectors of all element (shape [-1, 128])
        :param polyline_spans: vectors of i_th element is matrix[polyline_spans[i]]
        :return: hidden states of all elements and hidden states of lanes
        """

        raster_images = utils.get_from_mapping(mapping, 'image')
        raster_images = np.array(raster_images, dtype=np.float32)
        raster_images = torch.tensor(raster_images, device=device, dtype=torch.float32)
        # print(raster_images.shape)
        raster_images = raster_images.permute(0, 3, 1, 2).contiguous()
        args.raster_image_hidden = self.cnn_encoder(raster_images)


        input_list_list = []
        # input_list_list includes map data, this will be used in the future release.
        map_input_list_list = []
        lane_states_batch = None
        for i in range(batch_size):
            input_list = []
            map_input_list = []
            map_start_polyline_idx = mapping[i]['map_start_polyline_idx']
            for j, polyline_span in enumerate(polyline_spans[i]):
                tensor = torch.tensor(matrix[i][polyline_span], device=device)
                input_list.append(tensor)
                if j >= map_start_polyline_idx:
                    map_input_list.append(tensor)

            input_list_list.append(input_list)
            map_input_list_list.append(map_input_list)

        element_states_batch = utils.merge_tensors_not_add_dim(input_list_list, module=self.sub_graph,
                                                               sub_batch_size=16, device=device)

        inputs_before_laneGCN, inputs_lengths_before_laneGCN = utils.merge_tensors(element_states_batch, device=device)
        for i in range(batch_size):
            map_start_polyline_idx = mapping[i]['map_start_polyline_idx']
            agents = element_states_batch[i][:map_start_polyline_idx]
            lanes = element_states_batch[i][map_start_polyline_idx:]
            lanes = lanes + self.laneGCN_A2L(lanes.unsqueeze(0), torch.cat([lanes, agents[0:1]]).unsqueeze(0)).squeeze(0)

            element_states_batch[i] = torch.cat([agents, lanes])

        return element_states_batch, lane_states_batch

    # @profile
    def forward(self, mapping: List[Dict], device):
        import time
        global starttime
        starttime = time.time()

        matrix = utils.get_from_mapping(mapping, 'matrix')
        # TODO(cyrushx): Can you explain the structure of polyline spans?
        # vectors of i_th element is matrix[polyline_spans[i]]
        polyline_spans = utils.get_from_mapping(mapping, 'polyline_spans')

        batch_size = len(matrix)
        # for i in range(batch_size):
        # polyline_spans[i] = [slice(polyline_span[0], polyline_span[1]) for polyline_span in polyline_spans[i]]

        element_states_batch, lane_states_batch = self.forward_encode_sub_graph(mapping, matrix, polyline_spans, device, batch_size)

        inputs, inputs_lengths = utils.merge_tensors(element_states_batch, device=device)
        max_poly_num = max(inputs_lengths)
        attention_mask = torch.zeros([batch_size, max_poly_num, max_poly_num], device=device)
        for i, length in enumerate(inputs_lengths):
            attention_mask[i][:length][:length].fill_(1)

        hidden_states = self.global_graph(inputs, attention_mask, mapping)

        # utils.logging('time3', round(time.time() - starttime, 2), 'secs')

        return self.decoder(mapping, batch_size, lane_states_batch, inputs, inputs_lengths, hidden_states, device)
