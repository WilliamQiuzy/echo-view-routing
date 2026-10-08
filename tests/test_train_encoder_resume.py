"""Resuming the B0 trainers: the optimizer must be bound to the loaded model, so resumed training updates the loaded
weights (the earlier train() kept the optimizer of the discarded fresh model)."""
import torch
import torch.nn.functional as F

from echo_routing.features.encoder import FrameEncoder, save_checkpoint
from echo_routing.features.train_encoder import TrainConfig, resume_state


def test_resumed_optimizer_updates_the_loaded_model(tmp_path):
    cfg = TrainConfig.from_dict({"task": "family5", "epochs": 4, "lr": 1e-2})
    torch.manual_seed(0); model = FrameEncoder(2, "resnet18", pretrained=False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    x, y = torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 0, 1])
    F.cross_entropy(model(x), y).backward(); opt.step(); sched.step()
    path = save_checkpoint(tmp_path / "last.pt", model, {"epoch": 0, "optimizer": opt.state_dict(), "scheduler": sched.state_dict()})
    loaded, opt2, sched2, meta = resume_state(path, torch.device("cpu"), cfg)
    params = {id(p) for p in loaded.parameters()}
    assert all(id(p) in params for g in opt2.param_groups for p in g["params"]) and meta["epoch"] == 0
    assert sched2.last_epoch == 1 and opt2.state_dict()["state"]                         # optimizer and scheduler state restored
    before = loaded.head.weight.detach().clone(); loaded.train()
    opt2.zero_grad(); F.cross_entropy(loaded(x), y).backward(); opt2.step()
    assert not torch.equal(before, loaded.head.weight.detach())                          # the loaded weights move
