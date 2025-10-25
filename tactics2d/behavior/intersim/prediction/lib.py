import math

import numpy as np
import prediction.utils as utils
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LayerNorm(nn.Module):
    """
    This class conducts the layer normalization.
    It normalizes the inputs across the last dimensional axis (by default, across features).

    Attributes:
        hidden_size (int): The number of features in the input tensor.
        eps (float): A small constant for numerical stability added to the variance.
        weight (torch.nn.Parameter): The learnable weights applying to normalized outputs.
        bias (torch.nn.Parameter): The learnable biases applying to normalized outputs.
        variance_epsilon (float): Epsilon used for the numerical stability of normal distribution.
    """

    def __init__(self, hidden_size, eps=1e-5):
        """
        Initializes the LayerNorm instance.

        Args:
            hidden_size (int): The size of the last dimension of input (features).
            eps (float, optional): A small constant added to the variance to maintain numerical stability. Defaults to 1e-5.
        """
        super().__init__()
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        """
        Applies layer normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, hidden_size, ...).

        Returns:
            torch.Tensor: The normalized tensor of the same shape as the input.
        """
        # Compute the mean and standard deviation over the last dimension.
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)

        # Normalize the input using the computed mean and standard deviation.
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)

        # Apply the learnable weights and biases.
        return self.weight * x + self.bias


class MLP(nn.Module):
    """
    This class defines a simple multi-layer perceptron (MLP) class.
    This implementation includes a linear layer followed by a layer normalization and a ReLU non-linearity.

    Attributes:
        hidden_size (int): The number of features in the hidden layer.
        out_features (int): The number of output features. Defaults to the same as hidden size.
        linear (torch.nn.Linear): The linear transformation layer for the input.
        layer_norm (LayerNorm): The layer normalization layer applied after the linear transformation.
    """

    def __init__(self, hidden_size, out_features=None):
        """
        This function constructs an MLP instance.

        Args:
            hidden_size (int): The number of input features.
            out_features (int, optional): The number of output features. Defaults to None, falls back to hidden_size if None.
        """
        super().__init__()
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states):
        """
        This function implements forward pass of the MLP.

        Args:
            hidden_states (torch.Tensor): The input tensor containing hidden states.

        Returns:
            torch.Tensor: The output tensor after applying the linear layer, normalization, and ReLU activation.
        """
        # Apply linear layer
        hidden_states = self.linear(hidden_states)

        # Apply layer normalization
        hidden_states = self.layer_norm(hidden_states)

        # Apply ReLU activation function
        hidden_states = torch.nn.functional.relu(hidden_states)

        return hidden_states


class GlobalGraph(nn.Module):
    """
    This class is a global graph module implementing a self-attention mechanism, which can be used to capture dependencies between
    different elements in the input sequence.

    Attributes:
        num_attention_heads (int): The number of attention heads.
        attention_head_size (int): The size of each attention head, calculated as hidden_size // num_attention_heads.
        all_head_size (int): The total size of all attention heads.
        num_qkv (int): The number of query, key, value sets, set to 1 by default.
        query (torch.nn.Linear): Linear layer for query in the attention mechanism.
        key (torch.nn.Linear): Linear layer for key in the attention mechanism.
        value (torch.nn.Linear): Linear layer for value in the attention mechanism.
    """

    def __init__(self, hidden_size, attention_head_size=None, num_attention_heads=1):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = (
            hidden_size // num_attention_heads
            if attention_head_size is None
            else attention_head_size
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.num_qkv = 1

        self.query = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)
        self.key = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)
        self.value = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)

    def get_extended_attention_mask(self, attention_mask):
        """
        This function converts the attention mask to a format suitable for the attention score computation.
        This masks out the positions where attention is not required.

        Args:
            attention_mask (torch.Tensor): Tensor with shape [batch_size, seq_length] indicating whether each position
               should attend (1 indicates attention, 0 no attention).

        Returns:
            torch.Tensor: Extended attention mask with shape [batch_size, 1, 1, seq_length].
        """
        # Expand the attention_mask to be the same size as the attention scores
        extended_attention_mask = attention_mask.unsqueeze(1)
        # Mask positions where attention is not required (1 means attend, 0 don't attend)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def transpose_for_scores(self, x):
        """
        This function transposes the input tensor to the format expected by the attention scoring function.

        Args:
            x (torch.Tensor): The tensor containing the query, key, or value layers. Shape:
              [batch_size, seq_length, all_head_size].

        Returns:
            torch.Tensor: The transposed tensor, Shape: [batch_size, num_attention_heads, seq_length, attention_head_size].
        """
        sz = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        sz = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # (batch, max_vector_num, head, head_size)
        x = x.view(*sz)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, mapping=None, return_scores=False):
        """
        This function performs the forward pass for the global graph layer, which includes query, key and value transformations,
        attention scoring, masking and attention probability computation.

        Args:
            hidden_states (torch.Tensor): The input tensor with shape [batch_size, seq_length, hidden_size].
            attention_mask (torch.Tensor, optional): The attention mask with shape [batch_size, seq_length]. Defaults to None.
            mapping (Not used, optional): Placeholder for potential future use. Defaults to None.
            return_scores (bool, optional): Whether to return the attention scores along with the context layer. Defaults to False.

        Returns:
            torch.Tensor: The context layer after applying self-attention and concatenation of heads.
             If return_scores is True, returns a tuple containing the context layer and attention scores.
        """
        # Apply linear transformations to get query, key and value
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = nn.functional.linear(hidden_states, self.key.weight)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2)
        )
        # print(attention_scores.shape, attention_mask.shape)
        if attention_mask is not None:
            attention_scores = attention_scores + self.get_extended_attention_mask(attention_mask)
        # if utils.args.attention_decay and utils.second_span:
        #     attention_scores[:, 0, 0, 0] = attention_scores[:, 0, 0, 0] - self.attention_decay
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # if utils.args.attention_decay and utils.second_span:
        if False:
            utils.logging(self.attention_decay, prob=0.01)
            value_layer = torch.cat(
                [value_layer[:, 0:1, 0:1, :] * self.attention_decay, value_layer[:, 0:1, 1:, :]],
                dim=2,
            )
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if return_scores:
            assert attention_probs.shape[1] == 1
            attention_probs = torch.squeeze(attention_probs, dim=1)
            assert len(attention_probs.shape) == 3
            return context_layer, attention_probs
        return context_layer


class CrossAttention(GlobalGraph):
    """
    The cross-attention mechanism extends the self-attention to interact with external inputs, where the query comes
    from one source and the key-value pairs come from another.

    Attributes:
        query (torch.nn.Linear): Linear layer for transforming the query input.
        key (torch.nn.Linear): Linear layer for transforming the key input.
        value (torch.nn.Linear): Linear layer for transforming the value input. Note, if key_hidden_size and
            query_hidden_size are equal, value will be the same as key layer.
        all_head_size (int): The total size of each attention head, same as in GlobalGraph.
        num_qkv (int): The number of query, key, value sets, same as in GlobalGraph.
    """

    def __init__(
        self,
        hidden_size,
        attention_head_size=None,
        num_attention_heads=1,
        key_hidden_size=None,
        query_hidden_size=None,
    ):
        """
        This function constructs a CrossAttention instance.

        Args:
            hidden_size (int): The size of the input features.
            attention_head_size (int, optional): The size of each attention head, defaults to hidden_size // num_attention_heads if not provided.
            num_attention_heads (int, optional): The number of attention heads, defaults to 1.
            key_hidden_size (int, optional): The size of features for key and value. If None, uses hidden_size.
            query_hidden_size (int, optional): The size of features for query. If None, uses hidden_size.
        """
        super().__init__(hidden_size, attention_head_size, num_attention_heads)
        # If specific sizes for query and key are provided, adjust the corresponding layers accordingly.

    def __init__(
        self,
        hidden_size,
        attention_head_size=None,
        num_attention_heads=1,
        key_hidden_size=None,
        query_hidden_size=None,
    ):
        super().__init__(hidden_size, attention_head_size, num_attention_heads)
        if query_hidden_size is not None:
            self.query = nn.Linear(query_hidden_size, self.all_head_size * self.num_qkv)
        if key_hidden_size is not None:
            self.key = nn.Linear(key_hidden_size, self.all_head_size * self.num_qkv)
            self.value = nn.Linear(key_hidden_size, self.all_head_size * self.num_qkv)

    def forward(
        self,
        hidden_states_query,
        hidden_states_key=None,
        attention_mask=None,
        mapping=None,
        return_scores=False,
    ):
        mixed_query_layer = self.query(hidden_states_query)
        """
        This function applies cross-attention to originate attention from the hidden_states_key to the hidden_states_query.

        Args:
            hidden_states_query (torch.Tensor): The query tensor of shape
                                                [batch_size, seq_length_query, hidden_size].
            hidden_states_key (torch.Tensor, optional): The key tensor, if provided, of shape
                                                       [batch_size, seq_length_key, hidden_size].
                                                       Defaults to None, which implies self-attention.
            attention_mask (torch.Tensor, optional): The attention mask tensor of shape
                                                    [batch_size, seq_length_query, seq_length_key].
            mapping (Not used, optional): Placeholder for potential future use. Defaults to None.
            return_scores (bool, optional): Whether to return the attention scores along with the context layer.
                                             False by default.

        Returns:
            torch.Tensor: The context layer with shape [batch_size, seq_length, all_head_size].
                          If return_scores is True, returns a tuple of (context_layer, attention_scores).
        """
        # Ensure that key and value layers correspond to either the key input or the query input
        # depending on which one is provided. If neither is provided, default to self-attention.
        mixed_key_layer = self.key(hidden_states_key)
        mixed_value_layer = self.value(hidden_states_key)

        # Transform query, key, and value using separate linear layers and transpose them for attention scoring
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Calculate the attention scores and apply optional attention mask
        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2)
        )
        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2)
        )
        if attention_mask is not None:
            assert (
                hidden_states_query.shape[1] == attention_mask.shape[1]
                and hidden_states_key.shape[1] == attention_mask.shape[2]
            )
            attention_scores = attention_scores + self.get_extended_attention_mask(attention_mask)

        # Apply the softmax function to get the attention probabilities
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # Apply the attention probabilities to the value layer
        context_layer = torch.matmul(attention_probs, value_layer)

        # Reformat the context layer to combine the last two dimensions and prepare for final output
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # If requested, also return the attention scores along with the context layer
        if return_scores:
            return context_layer, torch.squeeze(attention_probs, dim=1)
        return context_layer


class GlobalGraphRes(nn.Module):
    """
    This class is the residual connection wrapper for the GlobalGraph module that processes hidden states through two
    GlobalGraph instances and merges their outputs by concatenation. This module is useful for building deep networks
    where the addition or concatenation of outputs aims to mitigate the vanishing gradient problem.

    Attributes:
        global_graph (GlobalGraph): An instance of GlobalGraph for the first graph-based self-attention block.
        global_graph2 (GlobalGraph): An instance of GlobalGraph for the second graph-based self-attention block.
    """

    def __init__(self, hidden_size):
        """
        This function constructs the GlobalGraphRes module with two GlobalGraph instances.

        Args:
            hidden_size (int): The size of the input features, must be even to allow for equal splitting if only one attention head is used.
        """
        super().__init__()
        # Initialize two instances of GlobalGraph with half the number of attention heads by default.
        super().__init__()
        self.global_graph = GlobalGraph(hidden_size, hidden_size // 2)
        self.global_graph2 = GlobalGraph(hidden_size, hidden_size // 2)

    def forward(self, hidden_states, attention_mask=None, mapping=None):
        """
        This function processes the input hidden_states through two GlobalGraph instances, concatenates their outputs, and returns the result.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_length, hidden_size].
            attention_mask (torch.Tensor, optional): Tensor defining the attention mask for the GlobalGraph instances.
            mapping (Not used, optional): Placeholder for potential future use.

        Returns:
            torch.Tensor: Concatenated outputs of the two GlobalGraph instances.
        """
        hidden_states = torch.cat(
            [
                self.global_graph(hidden_states, attention_mask, mapping),
                self.global_graph2(hidden_states, attention_mask, mapping),
            ],
            dim=-1,
        )
        # hidden_states = self.global_graph(hidden_states, attention_mask, mapping) \
        #                 + self.global_graph2(hidden_states, attention_mask, mapping)
        hidden_states = torch.cat(
            [
                self.global_graph(hidden_states, attention_mask, mapping),
                self.global_graph2(hidden_states, attention_mask, mapping),
            ],
            dim=-1,
        )
        return hidden_states

        return hidden_states


class PointSubGraph(nn.Module):
    """
    This class is a module that encodes 2D goals conditioned on the state of a target agent within a batch.

    Attributes:
        layers (nn.ModuleList): A list of MLP layers used for processing 2D point data and agent state.
        hidden_size (int): The size of the hidden layers in the MLPs, and also the size used to match agent state encoding.
    """

    def __init__(self, hidden_size):
        """
        This function constructs the PointSubGraph module with a series of MLP layers.

        Args:
            hidden_size (int): The size of the hidden layers for each MLP in the ModuleList. This determines the dimensionality of the internal feature representations.
        """
        super().__init__()
        self.hidden_size = hidden_size
        # Create a ModuleList of MLP layers with decreasing output sizes, each designed to reduce the dimensionality by half.
        self.layers = nn.ModuleList(
            [
                MLP(2, hidden_size // 2),
                MLP(hidden_size, hidden_size // 2),
                MLP(hidden_size, hidden_size),
            ]
        )
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList(
            [
                MLP(2, hidden_size // 2),
                MLP(hidden_size, hidden_size // 2),
                MLP(hidden_size, hidden_size),
            ]
        )

    def forward(self, hidden_states: torch.Tensor, agent: torch.Tensor) -> torch.Tensor:
        """
        This function processes the 2D coordinates along with the corresponding agent state through the MLP layers.

        Args:
            hidden_states (torch.Tensor): A tensor of shape [predict_agent_num, point_num, 2] representing the 2D point coordinates conditioned on the agent.
            agent (torch.Tensor): A tensor of shape [predict_agent_num, hidden_size] representing the state of the target agent.

        Returns:
            torch.Tensor: The output tensor after processing through the MLP layers.
        """
        device = hidden_states.device
        predict_agent_num, point_num = hidden_states.shape[0], hidden_states.shape[1]
        hidden_size = self.hidden_size
        assert (agent.shape[0], agent.shape[1]) == (predict_agent_num, hidden_size)
        agent = (
            agent[:, : hidden_size // 2]
            .unsqueeze(1)
            .expand([predict_agent_num, point_num, hidden_size // 2])
        )
        for layer_index, layer in enumerate(self.layers):
            if layer_index == 0:
                hidden_states = layer(hidden_states)
            else:
                hidden_states = layer(torch.cat([hidden_states, agent], dim=-1))

        return hidden_states


class SubGraph(nn.Module):
    """
    This class is a subgraph module of VectorNet, which applies multiple MLP layers with fully connected neurons, layer normalization,
    and ReLU activation to process the input hidden states.

    Attributes:
        args (Namespace): Arguments containing configuration for the subgraph layer.
        layers (nn.ModuleList): A list of MLP layers, where each layer consists of a fully connected layer followed by
                                 layer normalization and ReLU activation.
        depth (int): The number of MLP layers within the subgraph module.
    """

    def __init__(self, hidden_size, depth=None):
        """
        This class constructs the subgraph module with a specified number of MLP layers.

        Args:
            args (Namespace): Configuration arguments for the subgraph.
            hidden_size (int): The size of the hidden state for each MLP layer.
            depth (int, optional): The number of layers in the subgraph. Defaults to args.sub_graph_depth if not provided.
        """
        super().__init__()

    def __init__(self, args, hidden_size, depth=None):
        super().__init__()
        self.args = args
        if depth is None:
            depth = 3
        self.layers = nn.ModuleList([MLP(hidden_size, hidden_size // 2) for _ in range(depth)])

    def forward(self, hidden_states: torch.Tensor, li_vector_num=None):
        """
        This function processes the hidden states through the subgraph MLP layers with attention masking.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, vector_num, hidden_size].
            li_vector_num (list of int, optional): A list containing the number of vectors for each batch element.
                                                  Defaults to None, which implies max_vector_num for all elements.

        Returns:
            torch.Tensor: The processed tensor after subgraph processing.
        """
        sub_graph_batch_size = hidden_states.shape[0]
        max_vector_num = hidden_states.shape[1]
        if li_vector_num is None:
            li_vector_num = [max_vector_num] * sub_graph_batch_size
        hidden_size = hidden_states.shape[2]
        device = hidden_states.device

        # utils.logging('subgraph', sub_graph_batch_size, max_vector_num, hidden_size, prob=0.001)

        attention_mask = torch.zeros(
            [sub_graph_batch_size, max_vector_num, hidden_size // 2], device=device
        )
        zeros = torch.zeros([hidden_size // 2], device=device)
        for i in range(sub_graph_batch_size):
            # assert li_vector_num[i] > 0
            attention_mask[i][li_vector_num[i] : max_vector_num].fill_(-10000.0)
        for layer_index, layer in enumerate(self.layers):
            new_hidden_states = torch.zeros(
                [sub_graph_batch_size, max_vector_num, hidden_size], device=device
            )

            encoded_hidden_states = layer(hidden_states)
            for j in range(max_vector_num):
                # prevent encoding j-th vector itself.
                attention_mask[:, j] += -10000.0
                max_hidden, _ = torch.max(encoded_hidden_states + attention_mask, dim=1)
                max_hidden = torch.max(max_hidden, zeros)
                attention_mask[:, j] += 10000.0
                new_hidden_states[:, j] = torch.cat(
                    (encoded_hidden_states[:, j], max_hidden), dim=-1
                )
            hidden_states = new_hidden_states
        return torch.max(hidden_states, dim=1)[0]
