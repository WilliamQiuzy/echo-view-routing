"""PW-P1 exploratory clinical-purpose learning; no A5C deployment rule."""
import numpy as np


def pair_plan(n=19, steps=600):
    rng = np.random.RandomState(4702)
    values = []
    while len(values) < steps * 2: values.extend(rng.permutation(n).tolist())
    return np.asarray(values[:steps * 2], np.int64).reshape(steps, 2)


def fit_labels(n=19):
    true = np.tile(np.array([0., 1.], np.float32), (n, 1))
    shuffled = true.copy()
    shuffled[np.random.RandomState(4701).permutation(n)[:9]] = [1., 0.]
    return {'L': true, 'S': shuffled}


def metrics(scores):
    x = np.asarray(scores, np.float64).reshape(-1)
    if len(x) % 2 or not len(x) or not np.isfinite(x).all(): raise ValueError('finite paired scores required')
    y = np.tile([0, 1], len(x) // 2)
    positive, negative = x[y == 1], x[y == 0]
    auc = np.mean((positive[:, None] > negative[None, :]) + .5 * (positive[:, None] == negative[None, :]))
    order = np.sum((positive > negative) + .5 * (positive == negative))
    correct = int(np.sum((x > 0) == y))
    return {'correct': correct, 'total': len(x), 'accuracy': correct / len(x), 'AUROC': float(auc),
            'pair_order_points': float(order), 'pairs': len(x) // 2}


def fit_geometry(x):
    x = np.asarray(x, np.float64)
    mean, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1
    z = np.column_stack([(x - mean) / scale, np.ones(len(x))])
    penalty = np.eye(z.shape[1]) * .01; penalty[-1, -1] = 0
    coef = np.linalg.solve(z.T @ z + penalty, z.T @ np.tile([-1., 1.], len(x) // 2))
    return {'mean': mean, 'scale': scale, 'coef': coef}


def geometry_scores(x, control):
    x = np.asarray(x, np.float64)
    return np.column_stack([(x - control['mean']) / control['scale'], np.ones(len(x))]) @ control['coef']


def make_model(state):
    import torch
    from torch import nn
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.ops.misc import FrozenBatchNorm2d

    class PurposeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=None, norm_layer=FrozenBatchNorm2d)
            for m in self.backbone.modules():
                if isinstance(m, FrozenBatchNorm2d): m.eps = 0.
            self.backbone.load_state_dict(state, strict=True)
            for n, p in self.backbone.named_parameters():
                p.requires_grad_(n.startswith('body.layer4.') or n.startswith('fpn.'))
            self.head = nn.Linear(1024, 1)
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
            self.eval()

        def features(self, images):
            outputs = self.backbone(images)
            return torch.cat([outputs[str(i)].mean((2, 3)) for i in range(4)], dim=1)

        def forward(self, images):
            return self.head(self.features(images)).flatten()

    return PurposeModel()
