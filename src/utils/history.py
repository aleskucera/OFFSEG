from dataclasses import dataclass, field


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    train_mIoU: list[float] = field(default_factory=list)
    val_mIoU: list[float] = field(default_factory=list)
    lrs: list[float] = field(default_factory=list)
