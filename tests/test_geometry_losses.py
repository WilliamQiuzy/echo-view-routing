"""InfoNCE masks and the tilt term of geometry pretraining."""
import math

import torch

from echo_routing.geometry import losses


def test_partners_only_for_same_plane_pairs():
    p = losses.partners(torch.tensor([True, False, True]))
    assert p.tolist() == [1, 0, -1, -1, 5, 4]


def test_info_nce_matches_hand_computation_and_excludes_self():
    torch.manual_seed(0); z = torch.randn(6, 8); partner = losses.partners(torch.tensor([True, False, True])); tau = 0.1
    zn = torch.nn.functional.normalize(z, dim=1); sim = zn @ zn.T / tau; want = []
    for i in (0, 1, 4, 5):
        others = [j for j in range(6) if j != i]                      # never the anchor itself; partner once
        want.append(-(sim[i, int(partner[i])] - torch.logsumexp(sim[i, others], 0)))
    assert torch.allclose(losses.info_nce(z, partner, tau), torch.stack(want).mean(), atol=1e-6)


def test_info_nce_rewards_matching_partners():
    base = torch.eye(4)                                               # four orthogonal directions
    z_good = torch.stack([base[0], base[0], base[1], base[2], base[3], base[3]])   # pair 0 and pair 2 partners agree
    z_bad = torch.stack([base[0], base[1], base[1], base[2], base[3], base[0]])
    p = losses.partners(torch.tensor([True, False, True]))
    good, bad = losses.info_nce(z_good, p, 0.1), losses.info_nce(z_bad, p, 0.1)
    assert good < 0.1 < bad
    # with self excluded, a perfect match leaves only the other rows in the denominator: about log(1 + 4 e^-10)
    assert abs(float(good) - math.log(1 + 4 * math.exp(-10))) < 1e-3


def test_info_nce_without_anchors_is_zero_with_gradient_path():
    z = torch.randn(4, 8, requires_grad=True)
    loss = losses.info_nce(z, losses.partners(torch.tensor([False, False])), 0.1)
    loss.backward(); assert float(loss) == 0.0 and z.grad is not None


def test_tilt_loss_uses_pairs_in_order():
    torch.manual_seed(1); head = losses.TiltHead(dim=8, hidden=16); z = torch.randn(6, 8)
    target = torch.tensor([15.0, 0.0, -7.5])
    pred = head(z[[0, 2, 4]], z[[1, 3, 5]])
    want = torch.nn.functional.smooth_l1_loss(pred, target / 15.0)
    assert torch.allclose(losses.tilt_loss(head, z, target), want)
