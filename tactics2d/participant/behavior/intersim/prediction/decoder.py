from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from prediction.lib import PointSubGraph, CrossAttention, MLP
import prediction.utils as utils
from typing import Any, Dict, List, NamedTuple, Tuple

import numpy as np
import prediction.utils as utils
import torch
import torch.nn.functional as F
from prediction.lib import MLP, CrossAttention, GlobalGraph, GlobalGraphRes, PointSubGraph
from torch import Tensor, nn


class DecoderRes(nn.Module):
    """
    This class defines a decoder module with residual connections that applies an MLP followed by a fully connected layer.

    Attributes:
        mlp (MLP): A multi-layer perceptron (MLP) module that processes the input hidden states.
        fc (torch.nn.Linear): A fully connected layer that maps the output of the MLP to the desired output features.
    """

    def __init__(self, hidden_size, out_features=60):
        """
        This function constructs the decoder with residual connections.

        Args:
            hidden_size (int): The size of the hidden state dimension.
            out_features (int, optional): The number of output features. Defaults to 60.
        """
        super(DecoderRes, self).__init__()
        self.mlp = MLP(hidden_size, hidden_size)  # MLP with the same size input and output
        self.fc = nn.Linear(hidden_size, out_features)  # Final fully connected layer
        super().__init__()
        self.mlp = MLP(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, out_features)

    def forward(self, hidden_states):
        """
        This function is the forward pass of the decoder.

        Args:
            hidden_states (torch.Tensor): The input tensor to the decoder.

        Returns:
            torch.Tensor: The output tensor after passing through the MLP and fully connected layer.
        """
        # Apply the MLP and add it to the original hidden states for residual learning
        hidden_states = hidden_states + self.mlp(hidden_states)
        # Apply the fully connected layer to obtain the final output
        hidden_states = self.fc(hidden_states)
        return hidden_states

class DecoderResCat(nn.Module):
    """
    This class defines a decoder module with residual connections that concatenates input features before processing them through an MLP.

    Attributes:
        mlp (MLP): A multi-layer perceptron processing module that transforms the input hidden states.
        fc (torch.nn.Linear): A fully connected layer that outputs the final result after processing.
    """

    def __init__(self, hidden_size, in_features, out_features=60):
        """
        This function constructs the decoder with residual connections and concatenation.

        Args:
            hidden_size (int): The size of the hidden state used by the MLP.
            in_features (int): The number of input features to concatenate with the hidden states before processing.
            out_features (int, optional): The number of output features. Defaults to 60.
        """
        super(DecoderResCat, self).__init__()
        self.mlp = MLP(in_features, hidden_size)  # MLP transforming the hidden states
        self.fc = nn.Linear(hidden_size + in_features, out_features)  # Fully connected layer for final output
        super().__init__()
        self.mlp = MLP(in_features, hidden_size)
        self.fc = nn.Linear(hidden_size + in_features, out_features)

    def forward(self, hidden_states):
        """
        This function is the forward pass of the decoder with concatenation of input features.

        Args:
            hidden_states (torch.Tensor): The input tensor to the decoder.

        Returns:
            torch.Tensor: The output tensor after concatenation, MLP processing, and fully connected layer.
        """
        # Concatenate the output of the MLP with the original hidden states
        hidden_states = torch.cat([hidden_states, self.mlp(hidden_states)], dim=-1)
        # Apply the fully connected layer to the concatenated features to get the final output
        hidden_states = self.fc(hidden_states)
        return hidden_states

# Assuming 'utils' and 'PointSubGraph', 'CrossAttention', 'DecoderRes', 'DecoderResCat' are defined elsewhere as this code relies on them.

class Decoder(nn.Module):
    """
    This class defines a decoder module that integrates various components including a decoder with residual connections, 
    goals 2D MLPs, cross-attention mechanisms, and a point subgraph.

    Attributes:
        future_frame_num (int): The number of frames to predict in the future.
        mode_num (int): The number of interaction modes for the predictions.
        decoder (DecoderRes): A decoder with residual connections that outputs a 2D vector.
        goals_2D_mlps (torch.nn.Sequential): A sequential container of MLP layers for 2D goal prediction.
        goals_2D_decoder (DecoderResCat): A decoder with residual and concatenation features for goal 2D prediction.
        goals_2D_cross_attention (CrossAttention): A cross-attention module for processing 2D goals.
        goals_2D_point_sub_graph (PointSubGraph): A subgraph module for point-based processing of 2D goals.
        tnt_cross_attention (CrossAttention): A cross-attention module for TNt predictions.
        tnt_decoder (DecoderResCat): A decoder with residual and concatenation features for TNt predictions.
        inf_r_decoder (DecoderResCat): A decoder with residual and concatenation features for interaction range predictions.
    """

    def __init__(self, vectornet):
        super(Decoder, self).__init__()
        hidden_size = 128
        self.future_frame_num = 80
    def __init__(self, args_: utils.Args, vectornet):
        super().__init__()
        global args
        args = args_
        hidden_size = args.hidden_size
        self.future_frame_num = args.future_frame_num
        self.mode_num = 6

        self.decoder = DecoderRes(hidden_size, out_features=2)
        self.goals_2D_mlps = nn.Sequential(MLP(2, hidden_size), MLP(hidden_size), MLP(hidden_size))
        # self.goals_2D_decoder = DecoderRes(hidden_size * 3, out_features=1)
        self.goals_2D_decoder = DecoderResCat(hidden_size, hidden_size * 3, out_features=1)
        self.goals_2D_cross_attention = CrossAttention(hidden_size)
        self.goals_2D_point_sub_graph = PointSubGraph(hidden_size)
        self.goals_2D_decoder = DecoderResCat(hidden_size, hidden_size * 4, out_features=1)
        self.tnt_cross_attention = CrossAttention(hidden_size)
        self.tnt_decoder = DecoderResCat(
            hidden_size, hidden_size * 3, out_features=self.future_frame_num * 2
        )
        self.inf_r_decoder = DecoderResCat(hidden_size, hidden_size * 2, out_features=2)

    def detect_all_inf(self, hidden_states, batch_size, mapping, loss, device):
        """
        This function detects interaction ranges and compute the loss for the predictions.

        Args:
            hidden_states (torch.Tensor): The hidden states tensor to pass through the decoder.
            batch_size (int): Number of samples in the batch.
            mapping (List[Dict]): A list of dictionaries holding scenario and element information.
            loss (torch.Tensor): The current loss tensor to accumulate the new loss.
            device (torch.device): The device on which to run the computation (CPU or GPU).

        Returns:
            torch.Tensor: Updated loss with the interaction range prediction loss added.
        """
        # Flatten the hidden states to a 2D tensor for processing and pass through the interaction range decoder.
        hidden = hidden_states[:, :2, :].view(batch_size, -1)
        confidences = self.inf_r_decoder(hidden)
        outputs = F.log_softmax(confidences, dim=-1)
        interaction_labels = utils.get_from_mapping(mapping, "inf_label")
        loss += F.nll_loss(
            outputs, torch.tensor(interaction_labels, dtype=torch.long, device=device)
        )
        argmax = torch.argmax(outputs, dim=-1)

        for i in range(batch_size):
            ok = argmax[i] == interaction_labels[i]
            utils.other_errors_put(f'interaction_label.{interaction_labels[i]}', float(ok))
            utils.other_errors_put('interaction_label.all', float(ok))
            utils.other_errors_put(f"interaction_label.{interaction_labels[i]}", float(ok))
            utils.other_errors_put("interaction_label.all", float(ok))
            # this blows gpu memory when testing
            # globals.pred_relations[bytes.decode(mapping[i]['scenario_id'])] = [int(argmax[i]), torch.tensor(outputs[i],
            #                                                                                                 dtype=torch.float16)]
            # globals.pred_relations[bytes.decode(mapping[i]['scenario_id'])] = int(argmax[i])

    def eval_all_inf(self, hidden_states, batch_size, mapping, loss, device):
        """
        This function evaluates interaction ranges and return the scores and labels.

        Args:
            hidden_states (torch.Tensor): The hidden states tensor to process.
            batch_size (int): Number of samples in the batch.
            mapping (List[Dict]): A list of dictionaries holding additional related mappings.
            loss (torch.Tensor): The current loss tensor.
            device (torch.device): The device on which to perform the computation.

        Returns:
            Tuple of:
            - np.ndarray: Scores from the interaction range decoder.
            - List: All agent IDs from the mapping.
            - List: Scenario IDs from the mapping.
        """
        hidden = hidden_states[:, :2, :].view(batch_size, -1)
        confidences = self.inf_r_decoder(hidden)
        outputs = F.log_softmax(confidences, dim=-1)
        scores = torch.exp(outputs)
        all_agent_ids = utils.get_from_mapping(mapping, "all_agent_ids")
        scenario_ids = utils.get_from_mapping(mapping, "scenario_id")
        return scores.cpu().detach().numpy(), all_agent_ids, scenario_ids

    def forward(
        self,
        mapping: List[Dict],
        batch_size,
        lane_states_batch: List[Tensor],
        inputs: Tensor,
        inputs_lengths: List[int],
        hidden_states: Tensor,
        device,
    ):
        """
        This function is the forward pass through the decoder module to process hidden states for various predictions.

        Args:
            mapping (List[Dict]): A list of mappings with scenario and element related information.
            batch_size (int): The size of the batch.
            lane_states_batch (List[torch.Tensor]): A list containing hidden states for each lane.
            inputs (torch.Tensor): Hidden states of all elements before encoding by global graph.
            inputs_lengths (List[int]): List indicating valid number of elements for each sample in the batch.
            hidden_states (torch.Tensor): Hidden states after encoding by global graph.
            device (torch.device): The device on which to perform the computation.

        Returns:
            torch.Tensor: Tensor with computed losses from the decoder's predictions.
        """
        loss = torch.zeros(batch_size, device=device)
        # Initial call to evaluate all interactions based on input hidden states.
        return self.eval_all_inf(
            hidden_states=hidden_states,
            batch_size=batch_size,
            mapping=mapping,
            loss=loss,
            device=device,
        )
