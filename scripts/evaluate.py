# scripts/evaluate.py
import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any

from dwl_fallguard.detection import FallDetector
from dwl_fallguard.utils import load_model


def generate_realistic_test_data(num_samples: int = 1000, device: str = "cpu") -> tuple:
    """
    Generate test data with realistic falling vs. stable states.
    Mimics the logic used in train.py for consistency.
    """
    device = torch.device(device)
    
    # Base states
    states = torch.randn(num_samples, 42, device=device) * 0.2
    
    # 30% are falling
    is_falling = (torch.rand(num_samples, device=device) < 0.3)
    fall_indices = is_falling.nonzero(as_tuple=True)[0]
    
    if len(fall_indices) > 0:
        # Falling: high velocity + forward tilt
        states[fall_indices, 3:6] = torch.randn(len(fall_indices), 3, device=device) * 2.0
        states[fall_indices, 6:9] = torch.tensor([0.7, 0.3, 0.1], device=device).repeat(len(fall_indices), 1)
    
    labels = is_falling.long()
    return states, labels


def evaluate_fall_detection(
    model_path: Path,
    num_test_samples: int = 1000,
    device: str = "auto"
) -> Dict[str, Any]:
    """Comprehensive evaluation of fall detection on realistic synthetic data"""
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # Load model
    detector = load_model(FallDetector, model_path, state_dim=42, device=device)
    detector.eval()
    
    # Generate realistic test data
    test_states, true_labels = generate_realistic_test_data(num_test_samples, device=device)
    
    all_predictions = []
    all_uncertainties = []
    
    with torch.no_grad():
        batch_size = 64
        for i in range(0, num_test_samples, batch_size):
            batch = test_states[i:i+batch_size]
            predictions, uncertainties = detector(batch)
            
            # Extract fall probabilities (class 1)
            fall_probs = predictions[:, 1]
            all_predictions.extend(fall_probs.cpu().numpy())
            all_uncertainties.extend(uncertainties[:, 1].cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_uncertainties = np.array(all_uncertainties)
    true_labels_np = true_labels.cpu().numpy()
    
    # Evaluate at multiple thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = {}
    
    for threshold in thresholds:
        predicted_falls = all_predictions > threshold
        
        tp = np.sum((predicted_falls) & (true_labels_np == 1))
        fp = np.sum((predicted_falls) & (true_labels_np == 0))
        fn = np.sum((~predicted_falls) & (true_labels_np == 1))
        tn = np.sum((~predicted_falls) & (true_labels_np == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        results[f"threshold_{threshold:.1f}"] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "accuracy": float(accuracy),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn)
        }
    
    # Find best threshold by F1
    best_key = max(results.keys(), key=lambda k: results[k]["f1_score"])
    
    overall_results = {
        "best_f1_threshold": best_key,
        "best_f1_score": results[best_key]["f1_score"],
        "best_accuracy": results[best_key]["accuracy"],
        "mean_uncertainty_falls": float(all_uncertainties[true_labels_np == 1].mean()) if np.any(true_labels_np == 1) else 0.0,
        "mean_uncertainty_stable": float(all_uncertainties[true_labels_np == 0].mean()) if np.any(true_labels_np == 0) else 0.0,
        "total_samples": num_test_samples,
        "num_falls": int(true_labels_np.sum()),
        "detailed_results": results
    }
    
    # Save results
    output_path = Path("evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(overall_results, f, indent=2)
    
    # Print summary
    print("Evaluation Results:")
    print("=" * 60)
    print(f"Best F1 Score:     {overall_results['best_f1_score']:.4f} @ {best_key}")
    print(f"Best Accuracy:     {overall_results['best_accuracy']:.4f}")
    print(f"Mean Uncertainty (Falls):   {overall_results['mean_uncertainty_falls']:.4f}")
    print(f"Mean Uncertainty (Stable):  {overall_results['mean_uncertainty_stable']:.4f}")
    print(f"Total Test Samples: {num_test_samples} (Falls: {overall_results['num_falls']})")
    print(f"Results saved to: {output_path}")
    
    return overall_results


if __name__ == "__main__":
    results = evaluate_fall_detection(
        model_path=Path("models/fall_detector_final.pt"),
        num_test_samples=2000
    )