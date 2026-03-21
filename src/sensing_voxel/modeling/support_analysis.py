"""
Support requirement analysis for 3D printing.

Analyzes mesh to determine:
- Support material volume required
- Support contact area
- Overhang severity
- Bridge detection
- Critical support regions
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


@dataclass
class SupportRegion:
    """A region requiring support"""

    # Face indices requiring support
    face_indices: np.ndarray

    # Area of support region (mm²)
    area: float

    # Average angle from horizontal (degrees)
    avg_angle: float

    # Minimum Z height (mm from build plate)
    min_z: float

    # Maximum Z height
    max_z: float

    # Estimated support volume (mm³)
    estimated_volume: float

    # Is this a bridge (horizontal gap between supports)?
    is_bridge: bool = False


@dataclass
class SupportAnalysisResult:
    """Complete support analysis result"""

    success: bool

    # Total support volume required (mm³)
    total_support_volume: float = 0.0

    # Total contact area with model (mm²)
    total_contact_area: float = 0.0

    # List of support regions
    regions: list[SupportRegion] = field(default_factory=list)

    # Overhang statistics
    overhang_face_count: int = 0
    overhang_area: float = 0.0
    max_overhang_angle: float = 0.0

    # Bridge statistics
    bridge_count: int = 0
    bridge_area: float = 0.0
    longest_bridge: float = 0.0

    # Printability assessment
    printability_score: float = 1.0  # 0-1, higher is better
    is_printable: bool = True
    failure_reasons: list[str] = field(default_factory=list)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    errors: list[str] = field(default_factory=list)


@dataclass
class AnalysisConfig:
    """Configuration for support analysis"""

    # Angle from horizontal beyond which support is needed (degrees)
    overhang_threshold: float = 45.0

    # Maximum overhang angle before print likely fails (degrees)
    critical_overhang_threshold: float = 70.0

    # Minimum bridge length to report (mm)
    min_bridge_length: float = 5.0

    # Maximum bridge length that can be printed (mm)
    max_bridge_length: float = 50.0

    # Support contact threshold - minimum area to consider (mm²)
    min_support_area: float = 1.0

    # Layer height for support calculation (mm)
    layer_height: float = 0.2

    # Support density (ratio of solid material)
    support_density: float = 0.15

    # Print bed adhesion margin (mm)
    bed_margin: float = 10.0


class SupportAnalyzer:
    """
    Analyzes 3D models to determine support requirements.

    Provides detailed analysis of:
    - Overhangs requiring support
    - Bridges between supported regions
    - Support volume estimation
    - Printability assessment
    """

    def __init__(self, config: AnalysisConfig | None = None):
        self.config = config or AnalysisConfig()

    def analyze(
        self,
        mesh: trimesh.Trimesh | Path,
        base_normal: np.ndarray | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> SupportAnalysisResult:
        """
        Analyze mesh support requirements.

        Args:
            mesh: Trimesh object or path to mesh file
            base_normal: Orientation normal (default: Z-up)
            progress_callback: Optional callback(current, total)

        Returns:
            SupportAnalysisResult with complete analysis
        """
        result = SupportAnalysisResult(success=False)

        try:
            if isinstance(mesh, Path):
                mesh = trimesh.load_mesh(mesh, force="mesh")

            if mesh is None or len(mesh.vertices) == 0:
                result.errors.append("Empty or invalid mesh")
                return result

            # Orient mesh if base_normal provided
            if base_normal is not None:
                mesh = self._orient_mesh(mesh, base_normal)

            # Analyze overhangs
            overhang_data = self._analyze_overhangs(mesh)

            # Analyze bridges
            bridge_data = self._analyze_bridges(mesh, overhang_data)

            # Calculate support regions
            regions = self._calculate_support_regions(mesh, overhang_data)

            # Calculate totals
            total_volume = sum(r.estimated_volume for r in regions)
            total_area = sum(r.area for r in regions)

            # Assess printability
            printability, reasons, recommendations = self._assess_printability(
                mesh, overhang_data, bridge_data, regions
            )

            result.success = True
            result.total_support_volume = total_volume
            result.total_contact_area = total_area
            result.regions = regions
            result.overhang_face_count = overhang_data["face_count"]
            result.overhang_area = overhang_data["total_area"]
            result.max_overhang_angle = overhang_data["max_angle"]
            result.bridge_count = bridge_data["count"]
            result.bridge_area = bridge_data["total_area"]
            result.longest_bridge = bridge_data["max_length"]
            result.printability_score = printability
            result.is_printable = len(reasons) == 0
            result.failure_reasons = reasons
            result.recommendations = recommendations

        except Exception as e:
            logger.exception("Support analysis failed")
            result.errors.append(f"Analysis error: {e}")

        return result

    def _orient_mesh(
        self,
        mesh: trimesh.Trimesh,
        base_normal: np.ndarray,
    ) -> trimesh.Trimesh:
        """Rotate mesh so base_normal points down"""
        import math

        target = np.array([0, 0, -1], dtype=float)
        normal = np.array(base_normal, dtype=float)
        normal = normal / np.linalg.norm(normal)

        axis = np.cross(normal, target)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-10:
            if np.dot(normal, target) > 0:
                return mesh.copy()
            rotation = trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0])
        else:
            axis = axis / axis_norm
            angle = math.acos(np.clip(np.dot(normal, target), -1, 1))
            rotation = trimesh.transformations.rotation_matrix(angle, axis)

        oriented = mesh.copy()
        oriented.apply_transform(rotation)
        return oriented

    def _analyze_overhangs(self, mesh: trimesh.Trimesh) -> dict:
        """Analyze overhang geometry"""
        face_normals = mesh.face_normals
        face_areas = mesh.area_faces

        # Z component of normals
        z_normals = face_normals[:, 2]

        # Overhangs have normals pointing down (negative Z)
        overhang_mask = z_normals < 0

        # Calculate overhang angles from horizontal
        angles = np.degrees(np.arccos(np.clip(-z_normals, -1, 1)))
        angles[~overhang_mask] = 0

        # Filter by threshold
        significant_overhangs = angles > self.config.overhang_threshold

        overhang_face_count = np.sum(significant_overhangs)
        overhang_area = np.sum(face_areas[significant_overhangs])
        max_angle = np.max(angles) if overhang_face_count > 0 else 0

        return {
            "mask": significant_overhangs,
            "face_count": int(overhang_face_count),
            "total_area": float(overhang_area),
            "max_angle": float(max_angle),
            "angles": angles,
        }

    def _analyze_bridges(
        self,
        mesh: trimesh.Trimesh,
        overhang_data: dict,
    ) -> dict:
        """
        Detect bridge structures in the mesh.

        Bridges are horizontal gaps between supported regions.
        """
        overhang_mask = overhang_data["mask"]
        faces = mesh.faces
        vertices = mesh.vertices

        if not overhang_mask.any():
            return {"count": 0, "total_area": 0.0, "max_length": 0.0}

        # Group connected overhanging faces
        overhang_faces = np.where(overhang_mask)[0]

        # Simple bridge detection: horizontal faces at same Z level
        # with no support below
        bridge_count = 0
        bridge_area = 0.0
        max_bridge_length = 0.0

        for face_idx in overhang_faces:
            face = faces[face_idx]
            face_verts = vertices[face]

            # Check if face is roughly horizontal
            z_range = face_verts[:, 2].max() - face_verts[:, 2].min()
            if z_range > 0.5:  # Not horizontal
                continue

            avg_z = face_verts[:, 2].mean()

            # Check for support below (simplified: ray cast down)
            face_center = face_verts.mean(axis=0)
            ray_origin = face_center + np.array([0, 0, 0.01])
            ray_direction = np.array([0, 0, -1])

            # Check intersection with mesh
            locations, _, _ = mesh.ray.intersects_location(
                ray_origins=[ray_origin],
                ray_directions=[ray_direction],
            )

            if len(locations) == 0:
                # No support below - potential bridge
                bridge_count += 1
                bridge_area += mesh.area_faces[face_idx]

                # Estimate bridge length
                face_xy = face_verts[:, :2]
                bridge_length = np.linalg.norm(face_xy.max(axis=0) - face_xy.min(axis=0))
                max_bridge_length = max(max_bridge_length, bridge_length)

        return {
            "count": bridge_count,
            "total_area": float(bridge_area),
            "max_length": float(max_bridge_length),
        }

    def _calculate_support_regions(
        self,
        mesh: trimesh.Trimesh,
        overhang_data: dict,
    ) -> list[SupportRegion]:
        """Group overhanging faces into support regions"""
        faces = mesh.faces
        vertices = mesh.vertices
        face_areas = mesh.area_faces
        angles = overhang_data["angles"]
        overhang_mask = overhang_data["mask"]

        if not overhang_mask.any():
            return []

        regions = []
        visited = set()
        overhang_faces = np.where(overhang_mask)[0]

        # Adjacency: faces share an edge
        edges = {}
        for i, face in enumerate(faces):
            for j in range(3):
                edge = tuple(sorted([face[j], face[(j + 1) % 3]]))
                if edge not in edges:
                    edges[edge] = []
                edges[edge].append(i)

        # Build face adjacency
        face_adj = {i: set() for i in range(len(faces))}
        for edge_faces in edges.values():
            for f1 in edge_faces:
                for f2 in edge_faces:
                    if f1 != f2:
                        face_adj[f1].add(f2)
                        face_adj[f2].add(f1)

        for start_face in overhang_faces:
            if start_face in visited:
                continue

            # Flood fill to find connected region
            region_faces = []
            queue = [start_face]

            while queue:
                face_idx = queue.pop(0)
                if face_idx in visited:
                    continue

                visited.add(face_idx)
                region_faces.append(face_idx)

                # Add adjacent overhanging faces
                for adj in face_adj.get(face_idx, []):
                    if adj not in visited and overhang_mask[adj]:
                        queue.append(adj)

            if len(region_faces) < 1:
                continue

            # Calculate region properties
            region_face_indices = np.array(region_faces)
            area = sum(face_areas[i] for i in region_faces)

            if area < self.config.min_support_area:
                continue

            avg_angle = np.mean([angles[i] for i in region_faces])

            z_coords = [vertices[faces[i]].mean(axis=0)[2] for i in region_faces]
            min_z = min(z_coords)
            max_z = max(z_coords)

            # Estimate support volume
            support_height = max_z - min_z
            support_volume = area * support_height * self.config.support_density

            # Check if bridge
            is_bridge = avg_angle < 10 and support_height < 1.0

            region = SupportRegion(
                face_indices=region_face_indices,
                area=area,
                avg_angle=avg_angle,
                min_z=min_z,
                max_z=max_z,
                estimated_volume=support_volume,
                is_bridge=is_bridge,
            )
            regions.append(region)

        return regions

    def _assess_printability(
        self,
        mesh: trimesh.Trimesh,
        overhang_data: dict,
        bridge_data: dict,
        regions: list[SupportRegion],
    ) -> tuple[float, list[str], list[str]]:
        """
        Assess overall printability and provide recommendations.

        Returns:
            Tuple of (score, failure_reasons, recommendations)
        """
        score = 1.0
        failure_reasons = []
        recommendations = []

        # Check for critical overhangs
        if overhang_data["max_angle"] > self.config.critical_overhang_threshold:
            score -= 0.3
            failure_reasons.append(
                f"Overhang angle {overhang_data['max_angle']:.1f}° exceeds "
                f"safe threshold of {self.config.critical_overhang_threshold}°"
            )
            recommendations.append("Consider splitting the model or adding custom supports")

        # Check for excessive bridges
        if bridge_data["max_length"] > self.config.max_bridge_length:
            score -= 0.3
            failure_reasons.append(
                f"Bridge length {bridge_data['max_length']:.1f}mm exceeds printer capability"
            )
            recommendations.append("Add support pillars for long bridges")

        # Check support volume ratio
        if mesh.is_watertight:
            model_volume = mesh.volume
            support_ratio = sum(r.estimated_volume for r in regions) / model_volume

            if support_ratio > 1.0:
                score -= 0.2
                recommendations.append(
                    "Support material exceeds model volume - consider reorienting"
                )

        # Check for islands (unsupported floating geometry)
        # Simplified: check if any region is too high without support
        for region in regions:
            if region.min_z > 20 and not any(
                r.max_z > region.min_z and r.area > region.area for r in regions
            ):
                score -= 0.1
                recommendations.append("Consider model orientation to reduce floating regions")
                break

        # Clamp score
        score = max(0.0, min(1.0, score))

        if score >= 0.8:
            recommendations.append("Model is well-suited for FDM printing")
        elif score >= 0.5:
            recommendations.append("Printable with supports, review critical areas")
        else:
            recommendations.append("Review geometry before printing")

        return score, failure_reasons, recommendations

    def get_support_estimate(
        self,
        mesh: trimesh.Trimesh | Path,
        material_cost_per_cm3: float = 0.05,
    ) -> dict:
        """
        Get quick support estimate with material cost.

        Args:
            mesh: Mesh to analyze
            material_cost_per_cm3: Material cost in currency units per cm³

        Returns:
            Dictionary with support metrics and cost estimate
        """
        result = self.analyze(mesh)

        if not result.success:
            return {
                "success": False,
                "errors": result.errors,
            }

        support_cm3 = result.total_support_volume / 1000  # mm³ to cm³
        material_cost = support_cm3 * material_cost_per_cm3

        return {
            "success": True,
            "support_volume_cm3": support_cm3,
            "support_volume_mm3": result.total_support_volume,
            "contact_area_mm2": result.total_contact_area,
            "material_cost": material_cost,
            "printability_score": result.printability_score,
            "is_printable": result.is_printable,
            "requires_supports": len(result.regions) > 0,
            "recommendations": result.recommendations,
        }


class PrintabilityGate:
    """
    Gate that validates mesh printability before processing.

    Rejects meshes that cannot be made structurally sound
    through repair and orientation optimization.
    """

    def __init__(
        self,
        min_printability_score: float = 0.3,
        analyzer: SupportAnalyzer | None = None,
    ):
        self.min_printability_score = min_printability_score
        self.analyzer = analyzer or SupportAnalyzer()

    def validate(
        self,
        mesh: trimesh.Trimesh | Path,
        base_normal: np.ndarray | None = None,
    ) -> tuple[bool, str]:
        """
        Validate mesh printability.

        Args:
            mesh: Mesh to validate
            base_normal: Orientation (if pre-determined)

        Returns:
            Tuple of (is_valid, reason_or_empty)
        """
        try:
            if isinstance(mesh, Path):
                if not mesh.exists():
                    return False, "Mesh file not found"
                mesh = trimesh.load_mesh(mesh, force="mesh")

            if mesh is None or len(mesh.vertices) < 3:
                return False, "Invalid or empty mesh"

            # Basic checks
            if not mesh.is_watertight:
                # Allow non-watertight if it can be repaired
                if len(mesh.faces) < 4:
                    return False, "Mesh has too few faces to be printable"

            # Analyze support requirements
            result = self.analyzer.analyze(mesh, base_normal)

            if not result.success:
                return False, f"Analysis failed: {result.errors[0]}"

            # Check printability score
            if result.printability_score < self.min_printability_score:
                return (
                    False,
                    f"Printability score {result.printability_score:.2f} below "
                    f"threshold {self.min_printability_score}",
                )

            # Check critical failures
            if result.failure_reasons:
                return False, result.failure_reasons[0]

            return True, ""

        except Exception as e:
            logger.exception("Printability validation failed")
            return False, f"Validation error: {e}"
