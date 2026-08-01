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

    # --- wave-2 upgrade knobs (pre-wired 2026-07-29) ---
    # These fields are declared up front, before the work that consumes them, so that
    # three parallel agents never have to edit this shared file at the same time --
    # the exact collision that produced three duplicate WS-4 implementations on
    # 2026-07-24. Every default below reproduces the pre-wave behavior exactly.
    # See docs/UPGRADE_PLAN.md.

    # WS-6 · training quality (consumed by src/train.py, pretrain.py, finetune.py)
    val_fraction: float = 0.0        # share of names held out for validation; 0 = no split
    early_stop_patience: int = 0     # stop after N epochs with no val improvement; 0 = never
    lr_schedule: str = "none"        # "none" | "plateau" | "cosine"
    lr_factor: float = 0.5           # plateau: multiply the LR by this when val stalls
    lr_min: float = 0.0              # floor for any schedule

    # WS-7 · decoding quality (consumed by src/sample.py, evaluate.py)
    top_k: int = 0                   # sample only from the k likeliest next chars; 0 = off
    top_p: float = 1.0               # nucleus: smallest set with cumulative prob >= p; 1.0 = off
    repetition_penalty: float = 1.0  # >1 discourages chars already emitted; 1.0 = off
    min_length: int = 2              # discard generated names shorter than this

    # --- wave-3 knobs (pre-wired 2026-08-01) ---
    # Same discipline as wave 2 above: every field a wave-3 lane will consume is declared
    # here *before* that lane starts, so no two parallel agents ever edit this file at the
    # same time. Every default below reproduces pre-wave-3 behavior exactly, and
    # `from_dict` already ignores unknown keys, so old checkpoints stay loadable.
    # See docs/WAVE3_PLAN.md.

    # WS-9 · architecture (consumed by src/model.py, src/arch/)
    arch: str = "lstm"               # "lstm" | "gru" | "transformer"
    num_heads: int = 4               # transformer only: self-attention heads
    ff_dim: int = 512                # transformer only: feed-forward inner width
    max_position: int = 64           # transformer only: longest position it can encode

    # WS-10 · training regimen (consumed by src/train.py, pretrain.py, finetune.py)
    seed_init: bool = False          # also seed *before* model construction; see STATUS.md
    weight_decay: float = 0.0        # AdamW-style L2; 0.0 = plain Adam, today's behavior
    label_smoothing: float = 0.0     # softens the next-char target; 0.0 = off
    warmup_epochs: int = 0           # linear LR warmup over the first N epochs; 0 = off

    # WS-12 · provenance, for the web gallery and eval reports
    dataset_label: str = ""          # human name of the dataset this was trained on
    dataset_path: str = ""           # the --data path it was trained from

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        # Ignore unknown keys so old checkpoints stay loadable as the config grows.
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)
