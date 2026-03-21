"""
Mesh orientation optimization for best printability.

Optimizes model orientation to:
- Minimize support material
- Minimize overhangs
- Minimize print time
- Improve surface quality on critical faces
"""

import asyncio
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


@dataclass
class OrientationResult:
    """Result of orientation optimization"""

    success: bool
    optimal_rotation: np.ndarray | None = None
    base_normal: np.ndarray | None = None
    score: float = 0.0
    support_volume: float = 0.0
    overhang_area: float = 0.0
    print_height: float = 0.0
    errors: list[str] = field(default_factory=list)
    iterations: int = 0


@dataclass
class OrientationConstraints:
    """Constraints for orientation optimization"""

    # Minimum face area to consider (avoids tiny triangles)
    min_face_area: float = 0.1

    # Overhang angle threshold (degrees from horizontal)
    overhang_threshold: float = 45.0

    # Support contact threshold (mm)
    support_contact_threshold: float = 0.5

    # Maximum number of orientations to evaluate
    max_evaluations: int = 100

    # Target precision (radians)
    target_precision: float = 0.1


class OrientationOptimizer:
    """
    Optimizes mesh orientation for FDM 3D printing.

    Uses a combination of:
    1. Face normal analysis for candidate orientations
    2. Genetic algorithm refinement
    3. Support volume estimation
    """

    def __init__(self, constraints: OrientationConstraints | None = None):
        self.constraints = constraints or OrientationConstraints()
        self._rng = np.random.RandomState(42)

    def _compute_face_areas(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Compute area of each face"""
        return mesh.area_faces

    def _compute_face_normals(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Get face normals (already normalized in trimesh)"""
        return mesh.face_normals

    def _compute_support_volume(
        self,
        mesh: trimesh.Trimesh,
        base_normal: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Estimate support volume and overhang area for a given orientation.

        Returns:
            Tuple of (support_volume, overhang_area, print_height)
        """
        # Transform mesh so base_normal points down (0, 0, -1)
        rotation = self._rotation_to_z(base_normal)
        transformed = mesh.copy()
        transformed.apply_transform(rotation)

        # Get transformed vertices
        vertices = transformed.vertices
        faces = transformed.faces

        # Find faces with normals pointing down (overhangs)
        face_normals = transformed.face_normals
        dot_z = face_normals[:, 2]  # Z component (pointing down is negative)

        # Overhang angle in degrees from horizontal
        angles = np.degrees(np.arccos(np.clip(-dot_z, -1, 1)))
        overhang_mask = angles > self.constraints.overhang_threshold

        # Compute overhang area
        face_areas = self._compute_face_areas(transformed)
        overhang_area = np.sum(face_areas[overhang_mask])

        # Estimate support volume (simplified ray casting)
        support_volume = 0.0

        if overhang_mask.any():
            # Get bounding box
            min_z = vertices[:, 2].min()
            max_z = vertices[:, 2].max()
            print_height = max_z - min_z

            # For each overhanging face, estimate support needed
            overhanging_faces = faces[overhang_mask]
            for face in overhanging_faces:
                face_verts = vertices[face]
                face_z = face_verts[:, 2].mean()

                # Support volume is roughly face area * height to base
                support_height = face_z - min_z
                if support_height > self.constraints.support_contact_threshold:
                    support_volume += face_areas[overhang_mask][0] * support_height * 0.5

        else:
            print_height = vertices[:, 2].max() - vertices[:, 2].min()

        return support_volume, overhang_area, print_height

    def _rotation_to_z(self, normal: np.ndarray) -> np.ndarray:
        """
        Compute rotation matrix that aligns normal with negative Z axis.

        Args:
            normal: Unit normal vector

        Returns:
            4x4 transformation matrix
        """
        # Target: negative Z axis
        target = np.array([0, 0, -1], dtype=float)

        # Normalize input
        normal = np.array(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)

        # Compute rotation axis (cross product)
        axis = np.cross(normal, target)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-10:
            # Already aligned or anti-aligned
            if np.dot(normal, target) > 0:
                return np.eye(4)  # Identity
            else:
                # 180 degree rotation around X axis
                return trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0])

        axis = axis / axis_norm
        angle = math.acos(np.clip(np.dot(normal, target), -1, 1))

        rotation = trimesh.transformations.rotation_matrix(angle, axis)
        return rotation

    def _evaluate_orientation(
        self,
        mesh: trimesh.Trimesh,
        base_normal: np.ndarray,
    ) -> float:
        """
        Score an orientation (lower is better).

        Factors:
        - Support volume (primary)
        - Overhang area (secondary)
        - Print height (tertiary - affects print time)
        """
        support_vol, overhang_area, print_height = self._compute_support_volume(mesh, base_normal)

        # Normalize by mesh volume for scale-invariant scoring
        mesh_vol = mesh.volume if mesh.is_watertight else mesh.bounding_box.volume
        if mesh_vol < 1e-10:
            mesh_vol = 1.0

        # Score components (lower is better)
        support_score = support_vol / mesh_vol
        overhang_score = overhang_area / mesh.area
        height_score = print_height / (mesh.bounding_box.extents.max() or 1.0)

        # Weighted combination
        score = (
            10.0 * support_score  # Support volume is most important
            + 5.0 * overhang_score  # Overhangs affect quality
            + 1.0 * height_score  # Height affects print time
        )

        return score

    def _generate_candidate_orientations(
        self,
        mesh: trimesh.Trimesh,
    ) -> list[np.ndarray]:
        """
        Generate candidate base orientations from face normals.

        Uses the largest faces as candidate orientations,
        plus some random variations.
        """
        candidates = []

        # Get face normals and areas
        normals = self._compute_face_normals(mesh)
        areas = self._compute_face_areas(mesh)

        # Sort by area (descending)
        sorted_indices = np.argsort(areas)[::-1]

        # Take top faces as candidates
        num_candidates = min(20, len(normals))
        for i in range(num_candidates):
            idx = sorted_indices[i]
            if areas[idx] < self.constraints.min_face_area:
                break

            normal = normals[idx]
            # Ensure normal points up
            if normal[2] < 0:
                normal = -normal
            candidates.append(normal)

        # Add some random perturbations of the best candidates
        if candidates:
            for _ in range(10):
                base = random.choice(candidates[:5])
                # Random rotation around X and Y axes
                angle_x = self._rng.uniform(-0.3, 0.3)  # ~17 degrees
                angle_y = self._rng.uniform(-0.3, 0.3)

                rot_x = trimesh.transformations.rotation_matrix(angle_x, [1, 0, 0])
                rot_y = trimesh.transformations.rotation_matrix(angle_y, [0, 1, 0])

                perturbed = np.dot(rot_x[:3, :3], base)
                perturbed = np.dot(rot_y[:3, :3], perturbed)
                perturbed = perturbed / np.linalg.norm(perturbed)
                candidates.append(perturbed)

        # Add canonical orientations
        candidates.extend(
            [
                np.array([0, 0, 1]),  # Z-up
                np.array([0, 1, 0]),  # Y-up
                np.array([1, 0, 0]),  # X-up
                np.array([0, 0, -1]),  # Z-down
            ]
        )

        return candidates

    def optimize(
        self,
        mesh: trimesh.Trimesh | Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> OrientationResult:
        """
        Find optimal orientation for printing.

        Args:
            mesh: Trimesh object or path to mesh file
            progress_callback: Optional callback(current, total)

        Returns:
            OrientationResult with optimal rotation and metrics
        """
        result = OrientationResult(success=False)

        try:
            # Load mesh if path provided
            if isinstance(mesh, Path):
                mesh = trimesh.load_mesh(mesh, force="mesh")

            if mesh is None or len(mesh.vertices) == 0:
                result.errors.append("Empty or invalid mesh")
                return result

            # Generate candidate orientations
            candidates = self._generate_candidate_orientations(mesh)

            if not candidates:
                result.errors.append("No valid orientations generated")
                return result

            # Evaluate all candidates
            best_score = float("inf")
            best_normal = None

            total = min(len(candidates), self.constraints.max_evaluations)

            for i, normal in enumerate(candidates[:total]):
                if progress_callback:
                    progress_callback(i + 1, total)

                score = self._evaluate_orientation(mesh, normal)

                if score < best_score:
                    best_score = score
                    best_normal = normal.copy()

                result.iterations += 1

            if best_normal is None:
                result.errors.append("Failed to find valid orientation")
                return result

            # Compute final metrics
            support_vol, overhang_area, print_height = self._compute_support_volume(
                mesh, best_normal
            )

            # Compute optimal rotation matrix
            rotation = self._rotation_to_z(best_normal)

            result.success = True
            result.optimal_rotation = rotation
            result.base_normal = best_normal
            result.score = best_score
            result.support_volume = support_vol
            result.overhang_area = overhang_area
            result.print_height = print_height

        except Exception as e:
            logger.exception("Orientation optimization failed")
            result.errors.append(f"Optimization error: {e}")

        return result

    def apply_orientation(
        self,
        mesh: trimesh.Trimesh,
        orientation: OrientationResult,
    ) -> trimesh.Trimesh:
        """
        Apply optimal orientation to mesh.

        Args:
            mesh: Original mesh
            orientation: Orientation result from optimize()

        Returns:
            Re-oriented mesh (copy)
        """
        if not orientation.success or orientation.optimal_rotation is None:
            return mesh.copy()

        oriented = mesh.copy()
        oriented.apply_transform(orientation.optimal_rotation)
        return oriented


class GeneticOrientationOptimizer(OrientationOptimizer):
    """
    Genetic algorithm-based orientation optimizer.

    More thorough search than the base class, but slower.
    """

    def __init__(
        self,
        constraints: OrientationConstraints | None = None,
        population_size: int = 30,
        generations: int = 50,
        mutation_rate: float = 0.2,
    ):
        super().__init__(constraints)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def _random_orientation(self) -> np.ndarray:
        """Generate random orientation (unit vector)"""
        # Random point on unit sphere
        phi = self._rng.uniform(0, 2 * math.pi)
        cos_theta = self._rng.uniform(-1, 1)
        sin_theta = math.sqrt(1 - cos_theta**2)

        x = sin_theta * math.cos(phi)
        y = sin_theta * math.sin(phi)
        z = cos_theta

        return np.array([x, y, z])

    def _mutate(self, orientation: np.ndarray) -> np.ndarray:
        """Mutate an orientation slightly"""
        if self._rng.random() > self.mutation_rate:
            return orientation

        # Small random rotation
        axis = self._random_orientation()
        axis = np.cross(orientation, axis)
        if np.linalg.norm(axis) < 1e-10:
            return orientation

        axis = axis / np.linalg.norm(axis)
        angle = self._rng.uniform(-0.3, 0.3)  # ~17 degrees max

        rotation = trimesh.transformations.rotation_matrix(angle, axis)
        mutated = np.dot(rotation[:3, :3], orientation)
        return mutated / np.linalg.norm(mutated)

    def _crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
    ) -> np.ndarray:
        """Create child orientation from two parents"""
        # Linear interpolation then normalize
        alpha = self._rng.random()
        child = alpha * parent1 + (1 - alpha) * parent2
        return child / np.linalg.norm(child)

    def optimize(
        self,
        mesh: trimesh.Trimesh | Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> OrientationResult:
        """
        Find optimal orientation using genetic algorithm.
        """
        result = OrientationResult(success=False)

        try:
            if isinstance(mesh, Path):
                mesh = trimesh.load_mesh(mesh, force="mesh")

            if mesh is None or len(mesh.vertices) == 0:
                result.errors.append("Empty or invalid mesh")
                return result

            # Initialize population with candidates from base class
            candidates = self._generate_candidate_orientations(mesh)
            population = candidates[: self.population_size]

            # Fill rest with random orientations
            while len(population) < self.population_size:
                population.append(self._random_orientation())

            best_score = float("inf")
            best_normal = None

            for gen in range(self.generations):
                if progress_callback:
                    progress_callback(gen + 1, self.generations)

                # Evaluate population
                scores = []
                for orientation in population:
                    score = self._evaluate_orientation(mesh, orientation)
                    scores.append((score, orientation))

                    if score < best_score:
                        best_score = score
                        best_normal = orientation.copy()

                result.iterations += len(population)

                # Sort by score
                scores.sort(key=lambda x: x[0])

                # Select top performers (elitism)
                elite_count = max(2, self.population_size // 5)
                new_population = [s[1] for s in scores[:elite_count]]

                # Generate offspring
                while len(new_population) < self.population_size:
                    # Tournament selection
                    tournament_size = 3
                    parent1_idx = min(
                        [self._rng.randint(0, len(scores) - 1) for _ in range(tournament_size)],
                        key=lambda i: scores[i][0],
                    )
                    parent2_idx = min(
                        [self._rng.randint(0, len(scores) - 1) for _ in range(tournament_size)],
                        key=lambda i: scores[i][0],
                    )

                    parent1 = scores[parent1_idx][1]
                    parent2 = scores[parent2_idx][1]

                    # Crossover and mutation
                    child = self._crossover(parent1, parent2)
                    child = self._mutate(child)
                    new_population.append(child)

                population = new_population

            if best_normal is None:
                result.errors.append("Failed to find valid orientation")
                return result

            # Compute final metrics
            support_vol, overhang_area, print_height = self._compute_support_volume(
                mesh, best_normal
            )

            rotation = self._rotation_to_z(best_normal)

            result.success = True
            result.optimal_rotation = rotation
            result.base_normal = best_normal
            result.score = best_score
            result.support_volume = support_vol
            result.overhang_area = overhang_area
            result.print_height = print_height

        except Exception as e:
            logger.exception("Genetic orientation optimization failed")
            result.errors.append(f"Optimization error: {e}")

        return result
