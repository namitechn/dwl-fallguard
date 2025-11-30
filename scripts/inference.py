# scripts/inference.py (FINAL VERSION)
import torch
import argparse
from pathlib import Path

from dwl_fallguard.detection import FallDetector
from dwl_fallguard.recovery import RecoveryAgent
from dwl_fallguard.diagnosis import DiagnosisEngine
from dwl_fallguard.utils import load_model

def run_inference(
    state: torch.Tensor,
    detector_path: str = "models/fall_detector_final.pt",
    device: str = "auto"
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    detector = load_model(FallDetector, detector_path, state_dim=42, device=device)
    recovery_agent = RecoveryAgent()
    diagnosis_engine = DiagnosisEngine()

    print("DWL-FallGuard Inference Pipeline")
    print("=" * 50)

    detection = detector.detect_fall(state.to(device))
    print(f"Fall Detected: {detection['is_fall']}")
    print(f"Fall Probability: {detection['fall_probability']:.3f}")
    print(f"Confidence: {detection['confidence']:.3f}")
    print(f"Uncertainty: {detection['uncertainty']:.3f}")

    if detection['is_fall']:
        fall_type = "forward_fall"
        diagnosis = diagnosis_engine.diagnose(fall_type, {})
        print(f"\nDiagnosis: {diagnosis['primary_cause']}")
        print(f"Recommended Recovery: {diagnosis['recommended_recovery']}")

        # Pass device to recovery agent
        recovery_action = recovery_agent.recover(
            state.to(device), diagnosis, detection['uncertainty'], device=device
        )
        print(f"\nRecovery Mode: {recovery_action['mode']}")
        print(f"MPC Weight: {recovery_action['mpc_weight']:.2f}")
        print(f"Action (first 6): {[f'{x:.3f}' for x in recovery_action['action'][:6]]}")

    print("\n" + "=" * 50)
    return {
        "detection": detection,
        "diagnosis": diagnosis if detection['is_fall'] else None,
        "recovery": recovery_action if detection['is_fall'] else None
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DWL-FallGuard Inference")
    parser.add_argument("--state", type=str, default=None,
                       help="State vector as JSON string")
    parser.add_argument("--model", type=str, default="models/fall_detector_final.pt",
                       help="Path to detector model")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to run inference on")

    args = parser.parse_args()

    if args.state is None:
        test_states = [
            torch.zeros(42),
            torch.tensor([0.0, 0.0, 0.8, 2.5, 0.0, 0.0, 0.7, 0.3, 0.1] + [0.0]*33)
        ]
    else:
        test_states = [torch.tensor(eval(args.state))]

    for i, state in enumerate(test_states):
        scenario = "Stable" if i == 0 else "Fall Detected"
        print(f"\nScenario {i+1}: {scenario}")
        run_inference(state, args.model, args.device)