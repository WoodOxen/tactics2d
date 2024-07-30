from typing import Dict, List, Tuple, NamedTuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from prediction.lib import PointSubGraph, GlobalGraphRes, CrossAttention, GlobalGraph, MLP
import prediction.utils as utils

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
