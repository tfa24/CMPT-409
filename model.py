import lightning.pytorch as pl
import torch
from opacus import PrivacyEngine
from torch import Tensor, nn
from torch.optim import SGD
from torchmetrics.classification import Accuracy

from adam import Adam
from utils import (
    disabled_opacus_hooks,
    get_lambda_max_H,
    get_lambda_max_P_inv_H,
    get_P_diag,
)


class Model(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        # Training parameters
        learning_rate: float,
        max_epochs: int,
        optimizer: str,
        # DP parameters
        use_dp: bool,
        target_epsilon: float,
        target_delta: float,
        max_grad_norm: float,
        dp_adam_bc: bool,
        # skip any dataloader parameters
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.num_classes = num_classes

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.optimizer = optimizer

        self.use_dp = use_dp
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.dp_adam_bc = dp_adam_bc

        if self.use_dp:
            self.privacy_engine = PrivacyEngine()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 10),
        )

        # Losses and metrics.
        self.train_loss = nn.CrossEntropyLoss()
        self.dp_train_loss = None
        self.valid_loss = nn.CrossEntropyLoss()

        self.train_acc_top_1 = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )
        self.train_acc_top_5 = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=5
        )
        self.valid_acc_top_1 = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )
        self.valid_acc_top_5 = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=5
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.use_dp:
            checkpoint["dp_accountant_state"] = (
                self.privacy_engine.accountant.state_dict()
            )
            sd = checkpoint["state_dict"]
            checkpoint["state_dict"] = {
                k.replace("net._module.", "net."): v for k, v in sd.items()
            }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        dp_state = checkpoint.get("dp_accountant_state")
        if dp_state is not None:
            self._dp_accountant_state = dp_state

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        params = [p for p in self.parameters() if p.requires_grad]
        opt = self.optimizers()

        sample, target = batch
        pred = self(sample)

        with disabled_opacus_hooks(self.net):
            pred_copy = pred.clone()
            loss = self.train_loss(pred_copy, target)
            self.log("Training Loss", loss)
            self.log(
                "Training Accuracy Top 1",
                self.train_acc_top_1(pred, target),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "Training Accuracy Top 5",
                self.train_acc_top_5(pred, target),
                on_step=False,
                on_epoch=True,
            )

            if (self.global_step + 1) % 50 == 0:
                g = torch.autograd.grad(
                    loss,
                    params,
                    create_graph=True,
                    retain_graph=True,
                )
                lambda_max = get_lambda_max_H(g, params)
                self.log("Sharpness", lambda_max)

                if self.optimizer == "adam" and self.global_step != 0:
                    P_diag = get_P_diag(opt, params)
                    lambda_max = get_lambda_max_P_inv_H(g, params, P_diag)
                    self.log("Preconditioned Sharpness", lambda_max)

        if self.use_dp:
            loss = self.dp_train_loss(pred, target)

        loss.backward()
        opt.step()
        opt.zero_grad()

        return None

    def on_train_epoch_end(self):
        if self.use_dp:
            eps = self.privacy_engine.get_epsilon(self.target_delta)
            self.log("DP Epsilon", eps, on_step=False, on_epoch=True)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        sample, target = batch
        pred = self(sample)
        loss = self.valid_loss(pred, target)

        self.log(
            "Validation Accuracy Top 1",
            self.valid_acc_top_1(pred, target),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "Validation Accuracy Top 5",
            self.valid_acc_top_5(pred, target),
            on_step=False,
            on_epoch=True,
        )
        self.log("Validation Loss", loss, on_epoch=True, on_step=False)

        return loss

    def configure_optimizers(self) -> dict:
        match self.optimizer:
            case "sgd":
                optimizer = SGD(self.parameters(), lr=self.learning_rate)
            case "adam":
                optimizer = Adam(self.parameters(), lr=self.learning_rate)
            case _:
                raise ValueError(f"Optimizer {self.optimizer} not supported.")

        if self.use_dp:
            self.trainer.fit_loop.setup_data()

            self.net, optimizer, self.dp_train_loss, _ = (
                self.privacy_engine.make_private_with_epsilon(
                    module=self.net,
                    optimizer=optimizer,
                    criterion=nn.CrossEntropyLoss(),
                    data_loader=self.trainer.train_dataloader,
                    target_epsilon=self.target_epsilon,
                    target_delta=self.target_delta,
                    max_grad_norm=self.max_grad_norm,
                    epochs=self.max_epochs,
                    poisson_sampling=False,
                    grad_sample_mode="ghost",
                )
            )

            if hasattr(self, "_dp_accountant_state"):
                self.privacy_engine.accountant.load_state_dict(
                    self._dp_accountant_state
                )
                del self._dp_accountant_state

            if self.optimizer == "adam" and self.dp_adam_bc:
                sigma = optimizer.noise_multiplier
                C = optimizer.max_grad_norm
                B = optimizer.expected_batch_size
                noise_var = (sigma * C / B) ** 2

                optimizer.original_optimizer.dp_bc = True
                optimizer.original_optimizer.dp_noise_var = noise_var

        return {"optimizer": optimizer}
