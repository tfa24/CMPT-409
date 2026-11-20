from dataclasses import dataclass


@dataclass
class Config:
    # Model architecture parameters
    num_classes: int = 10
    # Training parameters
    learning_rate: float = 0.1
    max_epochs: int = 100000
    optimizer: str = "sgd"
    # DP parameters
    use_dp: bool = False
    target_epsilon: float = 2.0
    target_delta: float = 1e-5
    max_grad_norm: float = 3.0
    dp_adam_bc: bool = False
    # Dataloader parameters
    train_subset_size: int = 5000
    train_batch_size: int = 5000
    eval_batch_size: int = 10000
