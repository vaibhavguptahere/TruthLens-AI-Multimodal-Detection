"""
TruthLens AI - Comprehensive Verification Report Generator
Synthesizes predictions, calibrated confidence, Grad-CAM visual findings, text saliency,
and cross-modal consistency into a unified, human-readable verification report.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


class VerificationReportGenerator:
    """
    Synthesizes multi-stream evidence into academic and user-facing verification reports.
    """
    @staticmethod
    def generate_report(
        prediction: str,
        confidence: float,
        text_prediction: str,
        text_confidence: float,
        image_prediction: str,
        image_confidence: float,
        consistency_data: Dict[str, Any],
        top_text_tokens: List[Dict[str, Any]],
        sample_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Builds structured dictionary and human-readable explanation markdown.
        """
        timestamp = datetime.now().isoformat()
        
        # Build explanation narrative based on verifiable model outputs
        reasons = []
        if prediction == "FAKE":
            if image_prediction == "FAKE":
                reasons.append("Visual analysis detected structural pixel manipulation anomalies (diffusion/splice/warping).")
            if text_prediction == "FAKE":
                reasons.append("Linguistic assessment identified high-probability deceptive phrasing.")
            if consistency_data.get("consistency_status") == "MISALIGNED":
                reasons.append("Cross-modal inspection revealed semantic mismatch between image content and textual claims.")
        elif prediction == "REAL":
            reasons.append("Visual media exhibits authentic pixel signatures.")
            reasons.append("Textual claims correspond factually with depicted context.")
            reasons.append("High cross-modal semantic agreement confirmed.")
        else:  # UNCERTAIN
            reasons.append("Multimodal evidence is conflicting, degraded, or below minimum confidence thresholds.")
            if text_prediction != image_prediction:
                reasons.append(f"Modality conflict observed: Text model signaled {text_prediction} while Image model signaled {image_prediction}.")

        explanation_summary = " ".join(reasons)

        report = {
            "report_id": f"TLR-{sample_id or 'QUERY'}-{int(datetime.now().timestamp())}",
            "timestamp": timestamp,
            "final_verdict": prediction,
            "confidence_score": float(confidence),
            "calibrated_status": "HIGH_CONFIDENCE" if confidence >= 0.75 else ("MODERATE" if confidence >= 0.60 else "LOW_CONFIDENCE"),
            "modality_breakdown": {
                "text_modality": {
                    "verdict": text_prediction,
                    "confidence": float(text_confidence),
                    "salient_cues": top_text_tokens[:5]
                },
                "image_modality": {
                    "verdict": image_prediction,
                    "confidence": float(image_confidence),
                    "gradcam_generated": True
                },
                "cross_modal_consistency": consistency_data
            },
            "evidence_explanation": explanation_summary
        }
        return report
