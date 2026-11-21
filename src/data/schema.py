"""
TruthLens AI - Metadata Schema and Validation Definitions
Strictly adheres to CSET346 coursework and AGENTS.md guidelines.
"""

from enum import Enum
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


class ImageAuthenticity(str, Enum):
    AUTHENTIC = "authentic"
    MANIPULATED = "manipulated"
    UNKNOWN = "unknown"


class TextClaimStatus(str, Enum):
    VERIFIABLE_TRUE = "verifiable_true"
    FABRICATED = "fabricated"
    MISLEADING_OR_OPINION = "misleading_or_opinion"
    UNKNOWN = "unknown"


class ImageTextConsistency(str, Enum):
    ALIGNED = "aligned"
    MISALIGNED = "misaligned"
    AMBIGUOUS = "ambiguous"


class ManipulationType(str, Enum):
    NONE = "none"
    DIFFUSION_GENERATED = "diffusion_generated"
    FACE_SWAP = "face_swap"
    PHOTOSHOP_SPLICE = "photoshop_splice"
    OUT_OF_CONTEXT = "out_of_context"
    AMBIGUOUS = "ambiguous"


class PairLabel(str, Enum):
    REAL = "REAL"
    FAKE = "FAKE"
    UNCERTAIN = "UNCERTAIN"


class SplitType(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MultimodalSampleMetadata(BaseModel):
    """
    Standard metadata record for every multimodal sample in TruthLens AI.
    Ensures complete decoupling of visual authenticity, textual credibility,
    and cross-modal consistency.
    """
    sample_id: str = Field(..., description="Unique sample identifier (e.g. TL-REAL-001)")
    image_rel_path: str = Field(..., description="OS-agnostic relative path to image from data root")
    text: str = Field(..., min_length=3, description="Accompanying caption or context text")
    
    # Decoupled multimodal metadata fields
    image_authenticity: ImageAuthenticity
    text_claim_status: TextClaimStatus
    image_text_consistency: ImageTextConsistency
    manipulation_type: ManipulationType
    
    # Final 3-class target label
    pair_label: PairLabel
    
    # Lineage and governance
    source_dataset: str = Field(..., description="Origin source or benchmark lineage")
    source_reference: str = Field(..., description="DOI, URL, or citation reference")
    license: str = Field(default="Research / Educational Use", description="Content license")
    split: SplitType = Field(default=SplitType.TRAIN, description="Assigned dataset split")
    notes: Optional[str] = Field(default="", description="Explanatory notes on sample characteristics")

    @field_validator("image_rel_path")
    @classmethod
    def validate_image_rel_path(cls, v: str) -> str:
        # Guarantee forward slash agnostic relative paths
        clean_path = v.replace("\\", "/")
        if clean_path.startswith("/") or clean_path.startswith("C:") or clean_path.startswith("D:"):
            raise ValueError(f"image_rel_path must be relative, got: {v}")
        return clean_path
