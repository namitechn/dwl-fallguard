from typing import Dict, List, Any


class DiagnosisEngine:
    """Knowledge-based diagnosis system"""

    def __init__(self):
        self.knowledge_base = {
            "forward_fall": {
                "causes": ["obstacle", "slip", "sudden_stop"],
                "recovery": ["step_back", "knee_bend", "arm_swing"]
            },
            "backward_fall": {
                "causes": ["push", "acceleration", "wind"],
                "recovery": ["step_forward", "lean_forward", "arm_reach"]
            },
            "side_fall": {
                "causes": ["uneven_ground", "lateral_force", "sensor_error"],
                "recovery": ["side_step", "hip_shift", "arm_balance"]
            }
        }

    def diagnose(self, fall_type: str, state: Dict[str, float]) -> Dict[str, Any]:
        """Diagnose fall cause and recovery strategy"""
        if fall_type not in self.knowledge_base:
            fall_type = "forward_fall"

        kb = self.knowledge_base[fall_type]

        # Rank causes by state evidence
        cause_scores = self._score_causes(kb["causes"], state)
        top_causes = sorted(cause_scores.items(), key=lambda x: x[1], reverse=True)

        diagnosis = {
            "fall_type": fall_type,
            "primary_cause": top_causes[0][0] if top_causes else "unknown",
            "likely_causes": [c[0] for c in top_causes[:2]],
            "recommended_recovery": kb["recovery"][:2],
            "confidence": 0.85,
            "severity": "medium"
        }

        return diagnosis

    def _score_causes(self, causes: List[str], state: Dict[str, float]) -> Dict[str, float]:
        """Score causes based on state evidence"""
        scores = {}
        for cause in causes:
            score = 0.0
            if cause == "slip" and state.get("ground_friction", 0) < 0.4:
                score += 0.4
            if cause == "obstacle" and state.get("distance_sensor", 0) < 0.3:
                score += 0.3
            if cause == "sudden_stop" and state.get("velocity_change", 0) > 2.0:
                score += 0.35
            scores[cause] = score
        return scores

    def generate_report(self, diagnosis: Dict[str, Any]) -> str:
        """Generate diagnostic report"""
        report = f"""
FALL DIAGNOSIS REPORT
====================
Fall Type: {diagnosis['fall_type']}
Primary Cause: {diagnosis['primary_cause']}
Confidence: {diagnosis['confidence']:.1%}
Recommended Actions: {', '.join(diagnosis['recommended_recovery'])}
Severity: {diagnosis['severity']}
====================
        """
        return report.strip()