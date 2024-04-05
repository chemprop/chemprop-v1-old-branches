from abc import abstractmethod

from torch import nn, Tensor

from chemprop.nn.utils import get_activation_function


class FFN(nn.Module):
    r"""A :class:`FFN` is a differentiable function
    :math:`f_\theta : \mathbb R^i \mapsto \mathbb R^o`"""
    input_dim: int
    output_dim: int

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        pass


class MLP(nn.Sequential, FFN):
    r"""An :class:`MLP` is an FFN that implements the following function:

    .. math::
        \mathbf h_0 &= \mathbf x\,\mathbf W^{(0)} + \mathbf b^{(0)} \\
        \mathbf h_l &= \mathtt{dropout} \left(
            \sigma \left(\,\mathbf h_{l-1}\,\mathbf W^{(l)} \right)
        \right) \\
        \mathbf h_L &= \mathbf h_{L-1} \mathbf W^{(l)} + \mathbf b^{(l)},

    where :math:`\mathbf x` is the input tensor, :math:`\mathbf W^{(l)}` is the learned weight matrix
    for the :math:`l`-th layer, :math:`\mathbf b^{(l)}` is the bias vector for the :math:`l`-th layer,
    :math:`\mathbf h^{(l)}` is the hidden representation at layer :math:`l`, :math:`\sigma` is the
    activation function, and :math:`L` is the number of layers.
    """

    @classmethod
    def build(
        cls,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        dropout = nn.Dropout(dropout)
        act = get_activation_function(activation)
        dims = [input_dim] + [hidden_dim] * n_layers + [output_dim]
        blocks = [nn.Sequential(nn.Linear(dims[0], dims[1]))]
        if len(dims) > 2:
            blocks.extend(
                [
                    nn.Sequential(act, dropout, nn.Linear(d1, d2))
                    for d1, d2 in zip(dims[1:-1], dims[2:])
                ]
            )

        return cls(*blocks)

    @property
    def input_dim(self) -> int:
        return self[0][-1].in_features

    @property
    def output_dim(self) -> int:
        return self[-1][-1].out_features