"""The window network: its shapes, its marker input and the two weight file formats."""
import pytest
import torch
from safetensors.torch import save_file

from echoview import model as md


@pytest.fixture(scope="module")
def network() -> md.WindowModel:
    torch.manual_seed(0)
    return md.WindowModel().eval()


def inputs(batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    frames = torch.randint(0, 256, (batch, md.N_SLOTS, md.SIZE, md.SIZE, 3), dtype=torch.uint8, generator=torch.Generator().manual_seed(1))
    return md.normalise(frames), md.marker_planes(torch.tensor([3] * batch))


def test_one_window_gives_five_logits_and_leaves_no_hook_behind(network):
    rgb, marker = inputs()
    with torch.no_grad():
        logits = network(rgb, marker)
    assert logits.shape == (1, len(md.CLASSES))
    assert len(network.encoder.conv_proj._forward_hooks) == 0


def test_the_marker_changes_the_answer(network):
    torch.manual_seed(2)
    torch.nn.init.normal_(network.marker_stem.weight, std=0.05)
    rgb, marker = inputs()
    with torch.no_grad():
        first, second = network(rgb, marker), network(rgb, md.marker_planes(torch.tensor([12])))
    assert not torch.allclose(first, second)


def test_inputs_that_do_not_fit_are_refused(network):
    rgb, marker = inputs()
    with pytest.raises(ValueError, match="do not fit"):
        network(rgb, marker[:, :, :8])
    with pytest.raises(ValueError, match="uint8"):
        md.normalise(torch.zeros(1, 16, 224, 224, 3))


def test_marker_planes_set_one_whole_plane():
    planes = md.marker_planes(torch.tensor([0, 15]))
    assert planes.shape == (2, 1, 16, 224, 224)
    assert planes[0, 0, 0].min() == 1 and planes[0, 0, 1:].max() == 0 and planes[1, 0, 15].min() == 1


def test_both_weight_formats_load_the_same_network(network, tmp_path):
    state = {key: value.contiguous() for key, value in network.state_dict().items()}
    torch.save(state, tmp_path / "final.pt")
    save_file(state, str(tmp_path / "model.safetensors"))
    from_pt = md.load_model(tmp_path / "final.pt", torch.device("cpu"))
    from_st = md.load_model(tmp_path / "model.safetensors", torch.device("cpu"))
    for key, value in from_pt.state_dict().items():
        assert torch.equal(value, from_st.state_dict()[key]), key
    assert not any(p.requires_grad for p in from_st.parameters()) and not from_st.training


def test_a_file_with_a_missing_tensor_is_refused(network, tmp_path):
    state = dict(network.state_dict())
    del state["classifier.bias"]
    torch.save(state, tmp_path / "broken.pt")
    with pytest.raises(RuntimeError, match=r"classifier\.bias"):
        md.load_model(tmp_path / "broken.pt", torch.device("cpu"))
