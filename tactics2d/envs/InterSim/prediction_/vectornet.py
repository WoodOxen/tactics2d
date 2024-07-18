from typing import Dict, List, Tuple
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import utils as utils

class DecoderRes(nn.Module):
    def __init__(self, hidden_size, out_features=60):
        super(DecoderRes, self).__init__()
        self.mlp = MLP(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, out_features)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.mlp(hidden_states)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class DecoderResCat(nn.Module):
    def __init__(self, hidden_size, in_features, out_features=60):
        super(DecoderResCat, self).__init__()
        self.mlp = MLP(in_features, hidden_size)
        self.fc = nn.Linear(hidden_size + in_features, out_features)

    def forward(self, hidden_states):
        hidden_states = torch.cat([hidden_states, self.mlp(hidden_states)], dim=-1)
        hidden_states = self.fc(hidden_states)
        return hidden_states

class LayerNorm(nn.Module):
    r"""
    Layer normalization.
    """

    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MLP(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = torch.nn.functional.relu(hidden_states)
        return hidden_states


class GlobalGraph(nn.Module):
    r"""
    Global graph

    It's actually a self-attention.
    """

    def __init__(self, hidden_size, attention_head_size=None, num_attention_heads=1):
        super(GlobalGraph, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads if attention_head_size is None else attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.num_qkv = 1

        self.query = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)
        self.key = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)
        self.value = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)

        # self.attention_decay = nn.Parameter(torch.ones(1) * 0.5)

    def get_extended_attention_mask(self, attention_mask):
        """
        1 in attention_mask stands for doing attention, 0 for not doing attention.

        After this function, 1 turns to 0, 0 turns to -10000.0

        Because the -10000.0 will be fed into softmax and -10000.0 can be thought as 0 in softmax.
        """
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def transpose_for_scores(self, x):
        sz = x.size()[:-1] + (self.num_attention_heads,
                              self.attention_head_size)
        # (batch, max_vector_num, head, head_size)
        x = x.view(*sz)
        # (batch, head, max_vector_num, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, mapping=None, return_scores=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = nn.functional.linear(hidden_states, self.key.weight)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        # print(attention_scores.shape, attention_mask.shape)
        if attention_mask is not None:
            attention_scores = attention_scores + self.get_extended_attention_mask(attention_mask)
        # if utils.args.attention_decay and utils.second_span:
        #     attention_scores[:, 0, 0, 0] = attention_scores[:, 0, 0, 0] - self.attention_decay
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # if utils.args.attention_decay and utils.second_span:
        if False:
            utils.logging(self.attention_decay, prob=0.01)
            value_layer = torch.cat([value_layer[:, 0:1, 0:1, :] * self.attention_decay, value_layer[:, 0:1, 1:, :]],
                                    dim=2)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if return_scores:
            assert attention_probs.shape[1] == 1
            attention_probs = torch.squeeze(attention_probs, dim=1)
            assert len(attention_probs.shape) == 3
            return context_layer, attention_probs
        return context_layer


class CrossAttention(GlobalGraph):
    def __init__(self, hidden_size, attention_head_size=None, num_attention_heads=1, key_hidden_size=None,
                 query_hidden_size=None):
        super(CrossAttention, self).__init__(hidden_size, attention_head_size, num_attention_heads)
        if query_hidden_size is not None:
            self.query = nn.Linear(query_hidden_size, self.all_head_size * self.num_qkv)
        if key_hidden_size is not None:
            self.key = nn.Linear(key_hidden_size, self.all_head_size * self.num_qkv)
            self.value = nn.Linear(key_hidden_size, self.all_head_size * self.num_qkv)

    def forward(self, hidden_states_query, hidden_states_key=None, attention_mask=None, mapping=None,
                return_scores=False):
        mixed_query_layer = self.query(hidden_states_query)
        mixed_key_layer = self.key(hidden_states_key)
        mixed_value_layer = self.value(hidden_states_key)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        if attention_mask is not None:
            assert hidden_states_query.shape[1] == attention_mask.shape[1] \
                   and hidden_states_key.shape[1] == attention_mask.shape[2]
            attention_scores = attention_scores + self.get_extended_attention_mask(attention_mask)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if return_scores:
            return context_layer, torch.squeeze(attention_probs, dim=1)
        return context_layer


class GlobalGraphRes(nn.Module):
    def __init__(self, hidden_size):
        super(GlobalGraphRes, self).__init__()
        self.global_graph = GlobalGraph(hidden_size, hidden_size // 2)
        self.global_graph2 = GlobalGraph(hidden_size, hidden_size // 2)

    def forward(self, hidden_states, attention_mask=None, mapping=None):
        # hidden_states = self.global_graph(hidden_states, attention_mask, mapping) \
        #                 + self.global_graph2(hidden_states, attention_mask, mapping)
        hidden_states = torch.cat([self.global_graph(hidden_states, attention_mask, mapping),
                                   self.global_graph2(hidden_states, attention_mask, mapping)], dim=-1)
        return hidden_states


class PointSubGraph(nn.Module):
    """
    Encode 2D goals conditioned on target agent
    """

    def __init__(self, hidden_size):
        super(PointSubGraph, self).__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList([MLP(2, hidden_size // 2),
                                     MLP(hidden_size, hidden_size // 2),
                                     MLP(hidden_size, hidden_size)])

    def forward(self, hidden_states: Tensor, agent: Tensor):
        device = hidden_states.device
        predict_agent_num, point_num = hidden_states.shape[0], hidden_states.shape[1]
        hidden_size = self.hidden_size
        assert (agent.shape[0], agent.shape[1]) == (predict_agent_num, hidden_size)
        agent = agent[:, :hidden_size // 2].unsqueeze(1).expand([predict_agent_num, point_num, hidden_size // 2])
        for layer_index, layer in enumerate(self.layers):
            if layer_index == 0:
                hidden_states = layer(hidden_states)
            else:
                hidden_states = layer(torch.cat([hidden_states, agent], dim=-1))

        return hidden_states


class SubGraph(nn.Module):
    r"""
    Sub graph of VectorNet.

    It has three MLPs, each mlp is a fully connected layer followed by layer normalization and ReLU
    """

    def __init__(self, args, hidden_size, depth=None):
        super(SubGraph, self).__init__()
        self.args = args
        if depth is None:
            depth = args.sub_graph_depth
        self.layers = nn.ModuleList([MLP(hidden_size, hidden_size // 2) for _ in range(depth)])

    def forward(self, hidden_states: torch.Tensor, li_vector_num=None):
        args = self.args
        sub_graph_batch_size = hidden_states.shape[0]
        max_vector_num = hidden_states.shape[1]
        if li_vector_num is None:
            li_vector_num = [max_vector_num] * sub_graph_batch_size
        hidden_size = hidden_states.shape[2]
        device = hidden_states.device

        # utils.logging('subgraph', sub_graph_batch_size, max_vector_num, hidden_size, prob=0.001)

        attention_mask = torch.zeros([sub_graph_batch_size, max_vector_num, hidden_size // 2],
                                     device=device)
        zeros = torch.zeros([hidden_size // 2], device=device)
        for i in range(sub_graph_batch_size):
            # assert li_vector_num[i] > 0
            attention_mask[i][li_vector_num[i]:max_vector_num].fill_(-10000.0)
        for layer_index, layer in enumerate(self.layers):
            new_hidden_states = torch.zeros([sub_graph_batch_size, max_vector_num, hidden_size],
                                            device=device)

            encoded_hidden_states = layer(hidden_states)
            for j in range(max_vector_num):
                # prevent encoding j-th vector itself.
                attention_mask[:, j] += -10000.0
                max_hidden, _ = torch.max(encoded_hidden_states + attention_mask, dim=1)
                max_hidden = torch.max(max_hidden, zeros)
                attention_mask[:, j] += 10000.0
                new_hidden_states[:, j] = torch.cat((encoded_hidden_states[:, j], max_hidden), dim=-1)
            hidden_states = new_hidden_states
        return torch.max(hidden_states, dim=1)[0]


class Decoder(nn.Module):
    def __init__(self, args_: utils.Args, vectornet):
        super(Decoder, self).__init__()
        global args
        args = args_
        hidden_size = args.hidden_size
        self.future_frame_num = args.future_frame_num
        self.mode_num = 6

        self.decoder = DecoderRes(hidden_size, out_features=2)
        self.goals_2D_mlps = nn.Sequential(
            MLP(2, hidden_size),
            MLP(hidden_size),
            MLP(hidden_size)
        )
        # self.goals_2D_decoder = DecoderRes(hidden_size * 3, out_features=1)
        self.goals_2D_decoder = DecoderResCat(hidden_size, hidden_size * 3, out_features=1)
        self.goals_2D_cross_attention = CrossAttention(hidden_size)
        self.goals_2D_point_sub_graph = PointSubGraph(hidden_size)
        self.goals_2D_decoder = DecoderResCat(hidden_size, hidden_size * 4, out_features=1)
        self.tnt_cross_attention = CrossAttention(hidden_size)
        self.tnt_decoder = DecoderResCat(hidden_size, hidden_size * 3, out_features=self.future_frame_num * 2)
        self.inf_r_decoder = DecoderResCat(hidden_size, hidden_size * 2, out_features=2)

    def detect_all_inf(self, hidden_states, batch_size, mapping, loss, device):
        # Compute inf pred loss
        hidden = hidden_states[:, :2, :].view(batch_size, -1)
        confidences = self.inf_r_decoder(hidden)
        outputs = F.log_softmax(confidences, dim=-1)
        interaction_labels = utils.get_from_mapping(mapping, 'inf_label')
        loss += F.nll_loss(outputs, torch.tensor(interaction_labels, dtype=torch.long, device=device))
        argmax = torch.argmax(outputs, dim=-1)

        for i in range(batch_size):
            ok = argmax[i] == interaction_labels[i]
            utils.other_errors_put(f'interaction_label.{interaction_labels[i]}', float(ok))
            utils.other_errors_put('interaction_label.all', float(ok))
            # this blows gpu memory when testing
            # globals.pred_relations[bytes.decode(mapping[i]['scenario_id'])] = [int(argmax[i]), torch.tensor(outputs[i],
            #                                                                                                 dtype=torch.float16)]
            # globals.pred_relations[bytes.decode(mapping[i]['scenario_id'])] = int(argmax[i])

    def eval_all_inf(self, hidden_states, batch_size, mapping, loss, device):
        # Compute inf pred loss
        hidden = hidden_states[:, :2, :].view(batch_size, -1)
        confidences = self.inf_r_decoder(hidden)
        outputs = F.log_softmax(confidences, dim=-1)
        scores = torch.exp(outputs)
        all_agent_ids = utils.get_from_mapping(mapping, 'all_agent_ids')
        scenario_ids = utils.get_from_mapping(mapping, 'scenario_id')
        return scores.cpu().detach().numpy(), all_agent_ids, scenario_ids

    def forward(self, mapping: List[Dict], batch_size, lane_states_batch: List[Tensor], inputs: Tensor,
                inputs_lengths: List[int], hidden_states: Tensor, device):
        """
        :param lane_states_batch: each value in list is hidden states of lanes (value shape ['lane num', hidden_size])
        :param inputs: hidden states of all elements before encoding by global graph (shape [batch_size, 'element num', hidden_size])
        :param inputs_lengths: valid element number of each example
        :param hidden_states: hidden states of all elements after encoding by global graph (shape [batch_size, 'element num', hidden_size])
        """
        loss = torch.zeros(batch_size, device=device)
        return self.eval_all_inf(hidden_states=hidden_states, batch_size=batch_size, mapping=mapping,
                                 loss=loss, device=device)

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
