import torch

from openpi.fields import seds
from openpi.models.vlds import StructuredVLDSModel
from openpi.models.vlds import decode_seds_field
from openpi.models.vlds import evaluate_seds_field
from openpi.models.vlds import vlds_loss
from openpi.models.vlds_config import VLDSConfig
from openpi.training.vlds_dataset import SyntheticVLDSDataset
from openpi.training.vlds_dataset import validate_vlds_batch


def test_decoded_field_is_stable_and_zero_at_goal():
    config = VLDSConfig()
    raw = torch.randn(2, config.field_components, config.field_factor_dim)
    goal = torch.randn(2, 3)
    physical = decode_seds_field(raw, config)
    symmetric = (physical["matrices"] + physical["matrices"].transpose(-1, -2)) / 2
    assert torch.linalg.eigvalsh(symmetric).max().item() < -config.stability_margin + 1e-6
    velocity = evaluate_seds_field(raw, goal, goal, config)
    torch.testing.assert_close(velocity, torch.zeros_like(velocity), atol=1e-6, rtol=1e-6)


def test_torch_field_matches_numpy_decoder():
    config = VLDSConfig()
    raw = torch.randn(2, config.field_components, config.field_factor_dim) * 0.2
    goal = torch.tensor([[0.1, -0.2, 0.3], [-0.2, 0.4, 0.1]], dtype=torch.float32)
    positions = torch.randn(2, 9, 3) * 0.05 + goal[:, None, :]
    torch_velocity = evaluate_seds_field(raw, goal, positions, config)
    for index in range(2):
        numpy_field = seds.PositionField(raw[index].detach().numpy(), goal[index].numpy())
        numpy_velocity = numpy_field.velocity(positions[index].numpy())
        torch.testing.assert_close(
            torch_velocity[index], torch.from_numpy(numpy_velocity).to(torch_velocity.dtype), atol=2e-6, rtol=2e-6
        )


def test_structured_model_and_masked_loss_are_finite():
    config = VLDSConfig()
    dataset = SyntheticVLDSDataset(size=4, config=config)
    batch = {key: torch.stack([dataset[index][key] for index in range(4)]) for key in dataset.data}
    validate_vlds_batch(batch, config)
    model = StructuredVLDSModel(config)
    prediction = model(batch["state_history"], batch["execution_context"], batch["language"])
    assert prediction.goal_position.shape == (4, config.horizon, 3)
    assert prediction.field_raw.shape == (4, config.field_components, config.field_factor_dim)
    assert prediction.handoff_logits.shape == (4, config.horizon, 4)
    loss, metrics = vlds_loss(prediction, batch, config)
    assert torch.isfinite(loss)
    assert metrics["field_valid_fraction"] == 1.0
