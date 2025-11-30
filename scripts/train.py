import torch
import torch.nn.functional as F
from pathlib import Path
import logging
from typing import Dict, Any

from dwl_fallguard.detection import FallDetector
from dwl_fallguard.world_model import DWLExtended
from dwl_fallguard.utils import save_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_fallguard(
    num_epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 3e-4
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    detector = FallDetector(state_dim=42, hidden_dim=256).to(device)
    world_model = DWLExtended(state_dim=42, action_dim=12).to(device)

    opt_detector = torch.optim.AdamW(detector.parameters(), lr=learning_rate, weight_decay=1e-5)
    opt_world = torch.optim.AdamW(world_model.parameters(), lr=learning_rate, weight_decay=1e-5)

    history = {"detector_loss": [], "world_loss": [], "epoch": []}

    for epoch in range(num_epochs):
        detector.train()
        world_model.train()
        total_detector_loss = 0.0
        total_world_loss = 0.0

        for batch in range(50):
            # Realistic synthetic states
            batch_states = torch.randn(batch_size, 42, device=device) * 0.2
            is_falling = (torch.rand(batch_size, device=device) < 0.3)
            fall_indices = is_falling.nonzero(as_tuple=True)[0]
            if len(fall_indices) > 0:
                batch_states[fall_indices, 3:6] = torch.randn(len(fall_indices), 3, device=device) * 2.0
                batch_states[fall_indices, 6:9] = torch.tensor([0.7, 0.3, 0.1], device=device).repeat(len(fall_indices), 1)
            fall_labels = is_falling.long()

            # Actions and next states
            batch_actions = torch.randn(batch_size, 12, device=device)
            zero_padding = torch.zeros(batch_size, 30, device=device)  # 42 - 12 = 30
            action_effect = torch.cat([zero_padding, batch_actions], dim=1)
            batch_next_states = batch_states + 0.5 * action_effect + torch.randn_like(batch_states) * 0.1

            # Detector training
            detector_pred, _ = detector(batch_states)
            detector_loss = F.cross_entropy(detector_pred, fall_labels)
            opt_detector.zero_grad()
            detector_loss.backward()
            torch.nn.utils.clip_grad_norm_(detector.parameters(), 1.0)
            opt_detector.step()

            # World model training
            world_pred, uncertainty = world_model(batch_states, batch_actions)
            world_loss, metrics = world_model.compute_loss(batch_states, batch_actions, batch_next_states)
            opt_world.zero_grad()
            world_loss.backward()
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)
            opt_world.step()

            total_detector_loss += detector_loss.item()
            total_world_loss += world_loss.item()

        avg_detector_loss = total_detector_loss / 50
        avg_world_loss = total_world_loss / 50
        history["detector_loss"].append(avg_detector_loss)
        history["world_loss"].append(avg_world_loss)
        history["epoch"].append(epoch)

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{num_epochs}: Detector Loss: {avg_detector_loss:.4f}, World Loss: {avg_world_loss:.4f}")

        if epoch % 20 == 0:
            save_model(detector, Path("models") / "fall_detector.pt")
            save_model(world_model, Path("models") / "world_model.pt")

    save_model(detector, Path("models") / "fall_detector_final.pt")
    save_model(world_model, Path("models") / "world_model_final.pt")
    logger.info("Training completed successfully")
    return history

if __name__ == "__main__":
    training_history = train_fallguard(num_epochs=100)
    print("Training finished. Models saved to 'models/' directory")