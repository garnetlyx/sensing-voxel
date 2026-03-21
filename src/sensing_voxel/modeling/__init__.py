"""
Image-to-3D modeling package for Sensing Voxel.

Provides abstraction layer for multiple AI-powered image-to-3D model APIs
including TripoSR, Meshy, and Tripo3D.

Also includes mesh repair pipeline with MeshLab/Blender integration,
orientation optimization, and support analysis.
"""

from sensing_voxel.modeling.image_to_3d import (
    ImageTo3DClient,
    ImageTo3DProvider,
    MeshyClient,
    Tripo3DClient,
    TripoSRClient,
)
from sensing_voxel.modeling.cache import ModelCache
from sensing_voxel.modeling.cost_tracker import CostTracker, GenerationCost
from sensing_voxel.modeling.repair import (
    BlenderExecutor,
    MeshLabExecutor,
    MeshRepairPipeline,
    RepairResult,
)
from sensing_voxel.modeling.orientation import (
    GeneticOrientationOptimizer,
    OrientationConstraints,
    OrientationOptimizer,
    OrientationResult,
)
from sensing_voxel.modeling.support_analysis import (
    AnalysisConfig,
    PrintabilityGate,
    SupportAnalyzer,
    SupportAnalysisResult,
    SupportRegion,
)
from sensing_voxel.modeling.preview import (
    ApprovalMessage,
    PrintEstimates,
    calculate_estimates,
    create_approval_keyboard,
    create_approval_message,
    format_duration,
    render_multi_angle,
    render_thumbnail,
)

__all__ = [
    "ImageTo3DClient",
    "ImageTo3DProvider",
    "TripoSRClient",
    "MeshyClient",
    "Tripo3DClient",
    "ModelCache",
    "CostTracker",
    "GenerationCost",
    # Mesh repair
    "MeshLabExecutor",
    "BlenderExecutor",
    "MeshRepairPipeline",
    "RepairResult",
    # Orientation
    "OrientationOptimizer",
    "GeneticOrientationOptimizer",
    "OrientationConstraints",
    "OrientationResult",
    # Support analysis
    "SupportAnalyzer",
    "AnalysisConfig",
    "SupportAnalysisResult",
    "SupportRegion",
    "PrintabilityGate",
    # Preview
    "PrintEstimates",
    "ApprovalMessage",
    "render_thumbnail",
    "render_multi_angle",
    "calculate_estimates",
    "create_approval_message",
    "create_approval_keyboard",
    "format_duration",
]
