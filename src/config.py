"""Central place for all hyperparameters.

Keeping these in one small dataclass makes it trivial to tweak the model and to
serialize the exact settings alongside a checkpoint, so sampling always rebuilds
the identical architecture that was trained.
"""

from dataclasses import dataclass, asdict


# Special tokens used to bracket every name. START tells the model "begin a name";
# END tells it "the name is finished." PAD fills out short names when batching.
PAD_TOKEN = "\x00"
START_TOKEN = "\x02"  # begins every training name
END_TOKEN = "\x03"    # the model learns to emit this to stop generating


@dataclass
class Config:
    # --- model architecture ---
    embedding_dim: int = 32     # size of each character's dense vector
    hidden_dim: int = 256       # LSTM hidden-state width (the model's "memory")
    num_layers: int = 2         # stacked LSTM layers
    dropout: float = 0.2        # regularization between LSTM layers

    # --- training ---
    epochs: int = 300
    batch_size: int = 32
    learning_rate: float = 3e-3
    grad_clip: float = 5.0      # clip gradients to keep RNN training stable
    sample_every: int = 25      # print live samples every N epochs
    seed: int = 1337

    # --- generation defaults ---
    temperature: float = 0.8    # the "creativity" knob (see docs/PLAN.md)
    max_length: int = 40        # hard cap so a runaway sample can't loop forever

    # --- dual-output (WS-4): name + numeric attribute, see src/train_dual.py ---
    dual_output: bool = False   # if True, the model also carries a value-regression head
    value_mean: float = 0.0     # z-score stats for denormalizing the value head's output
    value_std: float = 1.0
    value_label: str = ""       # human label for the attribute, e.g. "founding year"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        # Ignore unknown keys so old checkpoints stay loadable as the config grows.
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)
