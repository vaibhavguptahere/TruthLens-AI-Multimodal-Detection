"""
TruthLens AI - Labeling Protocol
Defines the deterministic, academically defensible rules for mapping multimodal
sub-signals to pair_label (REAL, FAKE, UNCERTAIN).
"""

from src.data.schema import (
    ImageAuthenticity,
    TextClaimStatus,
    ImageTextConsistency,
    PairLabel
)


def determine_pair_label(
    image_auth: ImageAuthenticity,
    text_status: TextClaimStatus,
    consistency: ImageTextConsistency
) -> PairLabel:
    """
    Deterministically computes pair_label according to academic multimodal verification rules.

    Rules:
    1. If the image is MANIPULATED (deepfake, diffusion, photoshop splice) -> FAKE.
    2. If the text claim is explicitly FABRICATED or MISLEADING and semantic consistency is MISALIGNED -> FAKE.
    3. If the image is AUTHENTIC, text is VERIFIABLE_TRUE, and consistency is ALIGNED -> REAL.
    4. If there is conflicting evidence (e.g. authentic image with misleading caption that is partially ambiguous)
       or evidence is UNKNOWN/AMBIGUOUS -> UNCERTAIN.
    """
    # Direct visual manipulation -> FAKE
    if image_auth == ImageAuthenticity.MANIPULATED:
        return PairLabel.FAKE

    # Fabricated cross-modal mismatch (e.g., authentic image weaponized with false claim) -> FAKE
    if text_status == TextClaimStatus.FABRICATED and consistency == ImageTextConsistency.MISALIGNED:
        return PairLabel.FAKE

    # Clean authentic alignment -> REAL
    if (
        image_auth == ImageAuthenticity.AUTHENTIC
        and text_status == TextClaimStatus.VERIFIABLE_TRUE
        and consistency == ImageTextConsistency.ALIGNED
    ):
        return PairLabel.REAL

    # Ambiguous, conflicting, or unverified signals -> UNCERTAIN
    return PairLabel.UNCERTAIN
