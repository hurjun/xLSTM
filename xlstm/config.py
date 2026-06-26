"""Configuration dataclasses for the xLSTM building blocks and model.

These configs keep the architecture fully declarative so that depth, width,
head count and the sLSTM/mLSTM block pattern can be swapped without touching
the module code.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class sLSTMConfig:
    """Hyper-parameters for a (multi-head) sLSTM cell.

    Attributes:
        input_size: Dimensionality of the input vectors ``x_t``.
        hidden_size: Total number of scalar memory cells (``num_heads * head_dim``).
        num_heads: Number of heads. The recurrent (memory-mixing) weight is
            block-diagonal across heads, so a head only mixes with itself.
        forget_gate: Activation used for the forget gate. ``"sigmoid"`` keeps
            the classic bounded gate; ``"exponential"`` uses ``exp`` (fully
            exponential gating). The input gate is always exponential.
        dropout: Dropout applied to the cell output ``h_t``.
    """

    input_size: int
    hidden_size: int = 128
    num_heads: int = 4
    forget_gate: str = "sigmoid"
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})."
            )
        if self.forget_gate not in ("sigmoid", "exponential"):
            raise ValueError("forget_gate must be 'sigmoid' or 'exponential'.")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


@dataclass
class mLSTMConfig:
    """Hyper-parameters for a (multi-head) mLSTM cell with matrix memory.

    Attributes:
        input_size: Dimensionality of the input vectors ``x_t``.
        hidden_size: Total value dimension (``num_heads * head_dim``). The
            matrix memory of each head has shape ``(head_dim, head_dim)``.
        num_heads: Number of independent matrix-memory heads.
        dropout: Dropout applied to the cell output ``h_t``.
    """

    input_size: int
    hidden_size: int = 128
    num_heads: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})."
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


@dataclass
class BlockConfig:
    """Configuration for a single pre-norm residual xLSTM block.

    Attributes:
        kind: ``"slstm"`` or ``"mlstm"``.
        embedding_dim: Model/residual stream width ``d``.
        num_heads: Heads used by the inner cell.
        proj_factor: Up-projection factor for the block's inner width.
        conv_kernel_size: Causal depthwise conv kernel size. ``0`` disables it.
        forget_gate: Forget-gate activation for sLSTM blocks (ignored by mLSTM).
        dropout: Dropout used inside the block.
    """

    kind: str
    embedding_dim: int
    num_heads: int = 4
    proj_factor: float = 2.0
    conv_kernel_size: int = 4
    forget_gate: str = "sigmoid"
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("slstm", "mlstm"):
            raise ValueError("BlockConfig.kind must be 'slstm' or 'mlstm'.")


@dataclass
class xLSTMConfig:
    """Configuration for a full stacked xLSTM model.

    Attributes:
        vocab_size: Number of input/output token classes.
        embedding_dim: Residual stream width ``d``.
        num_blocks: Number of stacked blocks.
        block_pattern: Per-block kinds. If ``None`` an alternating
            ``mlstm``/``slstm`` pattern of length ``num_blocks`` is used.
        num_heads: Default head count for every block.
        proj_factor: Default up-projection factor.
        conv_kernel_size: Default causal conv kernel size (``0`` disables).
        forget_gate: Forget-gate activation for sLSTM blocks.
        dropout: Global dropout rate.
        tie_weights: Tie the embedding and the output projection.
    """

    vocab_size: int
    embedding_dim: int = 128
    num_blocks: int = 4
    block_pattern: list[str] | None = None
    num_heads: int = 4
    proj_factor: float = 2.0
    conv_kernel_size: int = 4
    forget_gate: str = "sigmoid"
    dropout: float = 0.0
    tie_weights: bool = True
    blocks: list[BlockConfig] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.block_pattern is None:
            # Alternate mLSTM / sLSTM, starting from mLSTM.
            self.block_pattern = [
                "mlstm" if i % 2 == 0 else "slstm" for i in range(self.num_blocks)
            ]
        if len(self.block_pattern) != self.num_blocks:
            raise ValueError(
                "block_pattern length must equal num_blocks "
                f"({len(self.block_pattern)} != {self.num_blocks})."
            )
        self.blocks = [
            BlockConfig(
                kind=kind,
                embedding_dim=self.embedding_dim,
                num_heads=self.num_heads,
                proj_factor=self.proj_factor,
                conv_kernel_size=self.conv_kernel_size,
                forget_gate=self.forget_gate,
                dropout=self.dropout,
            )
            for kind in self.block_pattern
        ]
