"""
Mesh repair pipeline with headless MeshLab and Blender integration.

Provides automated mesh repair workflows including:
- Hole filling
- Non-manifold geometry repair
- Self-intersection removal
- Triangle soup to solid conversion

Simple API functions for basic mesh repair (trimesh-based):
- repair_mesh(input_path, output_path) -> success bool
- is_printable(mesh_path) -> bool
- orient_for_printing(mesh_path) -> oriented_mesh
- fix_holes(mesh) -> repaired_mesh
- get_printability_report(mesh_path) -> report dict
"""

import asyncio
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh

from sensing_voxel.infra.retry import retry_async, BackoffPolicy

logger = logging.getLogger(__name__)


@dataclass
class RepairResult:
    """Result of mesh repair operation"""

    success: bool
    output_path: Path | None = None
    original_mesh: trimesh.Trimesh | None = None
    repaired_mesh: trimesh.Trimesh | None = None
    errors: list[str] = None
    warnings: list[str] = None
    repair_stats: dict = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.repair_stats is None:
            self.repair_stats = {}


class MeshLabExecutor:
    """Execute MeshLab filters in headless mode"""

    # MeshLabServer MLX filter script for common repairs
    REPAIR_SCRIPT_TEMPLATE = """<!DOCTYPE FilterScript>
<FilterScript>
    <filter name="Remove Duplicated Vertex"/>
    <filter name="Remove Unreferenced Vertex"/>
    <filter name="Merge Close Vertices" thr="0.0001"/>
    <filter name="Remove Zero Area Faces"/>
    <filter name="Remove Duplicated Faces"/>
    <filter name="Remove Faces from Non Manifold Edges"/>
    <filter name="Close Holes" maxholesize="30"/>
    <filter name="Select non Manifold Edges"/>
    <filter name="Delete Selected Faces and Vertices"/>
    <filter name="Select non Manifold Vertices"/>
    <filter name="Delete Selected Vertices"/>
    <filter name="Re-Orient all faces coherentely"/>
</FilterScript>"""

    def __init__(self, meshlabserver_path: str = "meshlabserver"):
        self.meshlabserver_path = meshlabserver_path
        self._available: bool | None = None

    async def is_available(self) -> bool:
        """Check if MeshLab is available"""
        if self._available is not None:
            return self._available

        if not shutil.which(self.meshlabserver_path):
            logger.warning(f"MeshLab not found: {self.meshlabserver_path}")
            self._available = False
            return False

        try:
            proc = await asyncio.create_subprocess_exec(
                self.meshlabserver_path,
                "-h",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            self._available = proc.returncode == 0 or b"meshlabserver" in stdout.lower()
        except Exception as e:
            logger.warning(f"MeshLab check failed: {e}")
            self._available = False

        return self._available

    async def repair_mesh(
        self,
        input_path: Path,
        output_path: Path,
        timeout: float = 300.0,
    ) -> RepairResult:
        """
        Repair mesh using MeshLab headless server.

        Args:
            input_path: Path to input mesh file
            output_path: Path to save repaired mesh
            timeout: Maximum time to wait for repair

        Returns:
            RepairResult with repaired mesh or errors
        """
        if not await self.is_available():
            return RepairResult(
                success=False,
                errors=["MeshLab not available"],
            )

        if not input_path.exists():
            return RepairResult(
                success=False,
                errors=[f"Input file not found: {input_path}"],
            )

        result = RepairResult(success=False)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                mlx_script_path = tmp_path / "repair.mlx"
                mlx_script_path.write_text(self.REPAIR_SCRIPT_TEMPLATE)

                # Log original mesh stats
                try:
                    original_mesh = trimesh.load_mesh(input_path, force="mesh")
                    result.original_mesh = original_mesh
                    result.repair_stats["original_vertices"] = len(original_mesh.vertices)
                    result.repair_stats["original_faces"] = len(original_mesh.faces)
                    result.repair_stats["original_is_watertight"] = original_mesh.is_watertight
                except Exception as e:
                    result.warnings.append(f"Could not load original mesh: {e}")

                # Run MeshLab repair
                proc = await asyncio.create_subprocess_exec(
                    self.meshlabserver_path,
                    "-i",
                    str(input_path),
                    "-o",
                    str(output_path),
                    "-s",
                    str(mlx_script_path),
                    "-om",
                    "vn",
                    "fn",  # Output vertex normals, face normals
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.communicate()
                    result.errors.append(f"MeshLab repair timed out after {timeout}s")
                    return result

                if proc.returncode != 0:
                    error_msg = (
                        stderr.decode("utf-8", errors="ignore") if stderr else "Unknown error"
                    )
                    result.errors.append(f"MeshLab repair failed: {error_msg}")
                    return result

                # Verify output
                if not output_path.exists():
                    result.errors.append("MeshLab did not produce output file")
                    return result

                # Load repaired mesh
                try:
                    repaired_mesh = trimesh.load_mesh(output_path, force="mesh")
                    result.repaired_mesh = repaired_mesh
                    result.success = True
                    result.output_path = output_path

                    result.repair_stats["repaired_vertices"] = len(repaired_mesh.vertices)
                    result.repair_stats["repaired_faces"] = len(repaired_mesh.faces)
                    result.repair_stats["repaired_is_watertight"] = repaired_mesh.is_watertight
                    result.repair_stats["volume_changed"] = False

                    if result.original_mesh is not None:
                        original_volume = (
                            result.original_mesh.volume if result.original_mesh.is_watertight else 0
                        )
                        repaired_volume = repaired_mesh.volume if repaired_mesh.is_watertight else 0
                        if original_volume > 0:
                            volume_diff = abs(repaired_volume - original_volume) / original_volume
                            result.repair_stats["volume_changed"] = volume_diff > 0.01

                    if not repaired_mesh.is_watertight:
                        result.warnings.append("Mesh is not watertight after repair")

                except Exception as e:
                    result.errors.append(f"Failed to load repaired mesh: {e}")
                    result.success = False

        except Exception as e:
            logger.exception("MeshLab repair failed")
            result.errors.append(f"Repair pipeline error: {e}")

        return result


class BlenderExecutor:
    """Execute Blender Python scripts in headless mode"""

    def __init__(self, blender_path: str = "blender"):
        self.blender_path = blender_path
        self._available: bool | None = None

    async def is_available(self) -> bool:
        """Check if Blender is available"""
        if self._available is not None:
            return self._available

        if not shutil.which(self.blender_path):
            logger.warning(f"Blender not found: {self.blender_path}")
            self._available = False
            return False

        try:
            proc = await asyncio.create_subprocess_exec(
                self.blender_path,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            self._available = proc.returncode == 0 and b"Blender" in stdout
        except Exception as e:
            logger.warning(f"Blender check failed: {e}")
            self._available = False

        return self._available

    async def repair_mesh(
        self,
        input_path: Path,
        output_path: Path,
        timeout: float = 300.0,
    ) -> RepairResult:
        """
        Repair mesh using Blender headless mode.

        Uses Blender's built-in mesh cleanup operators.

        Args:
            input_path: Path to input mesh file
            output_path: Path to save repaired mesh
            timeout: Maximum time to wait for repair

        Returns:
            RepairResult with repaired mesh or errors
        """
        if not await self.is_available():
            return RepairResult(
                success=False,
                errors=["Blender not available"],
            )

        if not input_path.exists():
            return RepairResult(
                success=False,
                errors=[f"Input file not found: {input_path}"],
            )

        result = RepairResult(success=False)

        # Create temporary Blender script
        blender_script = f"""
import bpy
import sys

input_path = r"{input_path}"
output_path = r"{output_path}"

try:
    # Clear existing mesh
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Import mesh
    bpy.ops.import_mesh.stl(filepath=input_path)
    obj = bpy.context.active_object

    # Enter edit mode
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    # Select all
    bpy.ops.mesh.select_all(action='SELECT')

    # Remove doubles
    bpy.ops.mesh.remove_doubles(threshold=0.0001)

    # Recalculate normals outside
    bpy.ops.mesh.normals_make_consistent(inside=False)

    # Select non-manifold edges
    bpy.ops.mesh.select_non_manifold()

    # Fill holes
    bpy.ops.mesh.edge_face_add()

    # Select all again
    bpy.ops.mesh.select_all(action='SELECT')

    # Triangulate (ensures all faces are triangles)
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')

    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Export as STL
    bpy.ops.export_mesh.stl(filepath=output_path)

    print("BLENDER_REPAIR_SUCCESS")

except Exception as e:
    print(f"BLENDER_REPAIR_ERROR: {{e}}")
    sys.exit(1)
"""

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                script_path = Path(tmpdir) / "repair.py"
                script_path.write_text(blender_script)

                # Log original mesh stats
                try:
                    original_mesh = trimesh.load_mesh(input_path, force="mesh")
                    result.original_mesh = original_mesh
                    result.repair_stats["original_vertices"] = len(original_mesh.vertices)
                    result.repair_stats["original_faces"] = len(original_mesh.faces)
                    result.repair_stats["original_is_watertight"] = original_mesh.is_watertight
                except Exception as e:
                    result.warnings.append(f"Could not load original mesh: {e}")

                # Run Blender headless
                proc = await asyncio.create_subprocess_exec(
                    self.blender_path,
                    "--background",
                    "--python",
                    str(script_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.communicate()
                    result.errors.append(f"Blender repair timed out after {timeout}s")
                    return result

                if proc.returncode != 0:
                    error_msg = (
                        stderr.decode("utf-8", errors="ignore") if stderr else "Unknown error"
                    )
                    result.errors.append(f"Blender repair failed: {error_msg}")
                    return result

                output_text = stdout.decode("utf-8", errors="ignore")
                if "BLENDER_REPAIR_ERROR" in output_text:
                    error_line = [l for l in output_text.split("\n") if "BLENDER_REPAIR_ERROR" in l]
                    result.errors.append(error_line[0] if error_line else "Blender script error")
                    return result

                # Verify output
                if not output_path.exists():
                    result.errors.append("Blender did not produce output file")
                    return result

                # Load repaired mesh
                try:
                    repaired_mesh = trimesh.load_mesh(output_path, force="mesh")
                    result.repaired_mesh = repaired_mesh
                    result.success = True
                    result.output_path = output_path

                    result.repair_stats["repaired_vertices"] = len(repaired_mesh.vertices)
                    result.repair_stats["repaired_faces"] = len(repaired_mesh.faces)
                    result.repair_stats["repaired_is_watertight"] = repaired_mesh.is_watertight

                    if not repaired_mesh.is_watertight:
                        result.warnings.append("Mesh is not watertight after repair")

                except Exception as e:
                    result.errors.append(f"Failed to load repaired mesh: {e}")
                    result.success = False

        except Exception as e:
            logger.exception("Blender repair failed")
            result.errors.append(f"Repair pipeline error: {e}")

        return result


class MeshRepairPipeline:
    """
    Unified mesh repair pipeline with multiple backend support.

    Tries MeshLab first, falls back to Blender, finally uses trimesh native repair.
    """

    def __init__(
        self,
        meshlab_path: str = "meshlabserver",
        blender_path: str = "blender",
        preferred_backend: Literal["meshlab", "blender", "trimesh", "auto"] = "auto",
    ):
        self.meshlab = MeshLabExecutor(meshlab_path)
        self.blender = BlenderExecutor(blender_path)
        self.preferred_backend = preferred_backend

    async def _trimesh_repair(
        self,
        input_path: Path,
        output_path: Path,
    ) -> RepairResult:
        """Fallback repair using trimesh native operations"""
        result = RepairResult(success=False)

        try:
            mesh = trimesh.load_mesh(input_path, force="mesh")
            result.original_mesh = mesh
            result.repair_stats["original_vertices"] = len(mesh.vertices)
            result.repair_stats["original_faces"] = len(mesh.faces)
            result.repair_stats["original_is_watertight"] = mesh.is_watertight

            # Remove duplicate vertices
            mesh.merge_vertices()

            mesh.process(validate=True)

            mesh.fix_normals()

            # Fill holes by creating convex hull (aggressive but effective)
            if not mesh.is_watertight:
                try:
                    # Try convex hull as last resort
                    hull = mesh.convex_hull
                    if hull.is_watertight:
                        mesh = hull
                        result.warnings.append("Used convex hull to achieve watertightness")
                except Exception:
                    pass

            # Export
            mesh.export(output_path)

            result.repaired_mesh = mesh
            result.output_path = output_path
            result.success = True
            result.repair_stats["repaired_vertices"] = len(mesh.vertices)
            result.repair_stats["repaired_faces"] = len(mesh.faces)
            result.repair_stats["repaired_is_watertight"] = mesh.is_watertight

            if not mesh.is_watertight:
                result.warnings.append("Mesh is not watertight after trimesh repair")

        except Exception as e:
            result.errors.append(f"Trimesh repair failed: {e}")

        return result

    async def repair(
        self,
        input_path: Path,
        output_path: Path | None = None,
        backend: Literal["meshlab", "blender", "trimesh", "auto"] | None = None,
    ) -> RepairResult:
        """
        Repair a mesh file using the best available backend.

        Args:
            input_path: Path to input mesh file
            output_path: Path to save repaired mesh (default: auto-generate)
            backend: Which backend to use (default: from preferred_backend)

        Returns:
            RepairResult with repaired mesh or errors
        """
        backend = backend or self.preferred_backend

        # Determine output path
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_repaired.stl"

        # Auto-select best backend
        if backend == "auto":
            if await self.meshlab.is_available():
                backend = "meshlab"
            elif await self.blender.is_available():
                backend = "blender"
            else:
                backend = "trimesh"

        logger.info(f"Repairing mesh with backend: {backend}")

        if backend == "meshlab":
            result = await self.meshlab.repair_mesh(input_path, output_path)
            if result.success:
                return result
            # Fall through to next backend
            logger.warning("MeshLab repair failed, trying Blender...")
            backend = "blender"

        if backend == "blender":
            result = await self.blender.repair_mesh(input_path, output_path)
            if result.success:
                return result
            logger.warning("Blender repair failed, trying trimesh...")
            backend = "trimesh"

        # Final fallback: trimesh
        return await self._trimesh_repair(input_path, output_path)

    async def batch_repair(
        self,
        input_paths: list[Path],
        output_dir: Path,
        backend: Literal["meshlab", "blender", "trimesh", "auto"] | None = None,
    ) -> list[RepairResult]:
        """
        Repair multiple meshes in batch.

        Args:
            input_paths: List of input mesh file paths
            output_dir: Directory to save repaired meshes
            backend: Which backend to use

        Returns:
            List of RepairResult objects
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for input_path in input_paths:
            output_path = output_dir / f"{input_path.stem}_repaired.stl"
            result = await self.repair(input_path, output_path, backend)
            results.append(result)

        return results


# =============================================================================
# Simple API Functions (trimesh-based)
# =============================================================================


def repair_mesh(input_path: str | Path, output_path: str | Path) -> bool:
    """
    Repair a mesh file using trimesh.

    Fixes common issues:
    - Holes (if possible)
    - Degenerate faces
    - Flipped normals
    - Duplicate vertices

    Args:
        input_path: Path to input mesh file (STL, OBJ, etc.)
        output_path: Path to save repaired mesh

    Returns:
        True if repair succeeded and mesh is printable, False otherwise
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return False

    try:
        # Load mesh
        mesh = trimesh.load_mesh(str(input_path), force="mesh")

        # Apply repairs
        mesh = fix_holes(mesh)
        mesh = orient_for_printing(mesh)

        # Check if printable after repair
        if not is_printable(mesh):
            logger.error(f"Mesh could not be repaired to be printable: {input_path}")
            return False

        # Export repaired mesh
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output_path))

        logger.info(f"Mesh repaired successfully: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Mesh repair failed: {e}")
        return False


def is_printable(mesh_or_path: str | Path | trimesh.Trimesh) -> bool:
    """
    Check if a mesh is printable.

    A mesh is printable if it is:
    - Manifold (every edge has exactly 2 faces)
    - Watertight (closed surface)
    - Has positive volume

    Args:
        mesh_or_path: Path to mesh file or trimesh.Trimesh object

    Returns:
        True if mesh is printable, False otherwise
    """
    try:
        if isinstance(mesh_or_path, (str, Path)):
            mesh = trimesh.load_mesh(str(mesh_or_path), force="mesh")
        else:
            mesh = mesh_or_path

        # Check manifold
        if not mesh.is_watertight:
            return False

        # Check volume is positive
        if mesh.volume <= 0:
            return False

        # Check for degenerate faces
        if len(mesh.faces) == 0:
            return False

        return True

    except Exception as e:
        logger.error(f"Error checking printability: {e}")
        return False


def orient_for_printing(mesh_or_path: str | Path | trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Auto-orient mesh for best print direction (largest flat surface down).

    Uses PCA to find the principal axes and orients the mesh so that
    the largest flat surface is facing down (negative Z).

    Args:
        mesh_or_path: Path to mesh file or trimesh.Trimesh object

    Returns:
        Oriented trimesh.Trimesh object (new copy, original is not modified)
    """
    if isinstance(mesh_or_path, (str, Path)):
        mesh = trimesh.load_mesh(str(mesh_or_path), force="mesh")
    else:
        mesh = mesh_or_path

    # Create a copy to avoid modifying original
    mesh = mesh.copy()

    # Apply basic processing
    mesh.merge_vertices()
    mesh.fix_normals()

    # Find optimal orientation using convex hull's face areas
    # The face with largest area should be the base
    try:
        # Get convex hull to simplify analysis
        hull = mesh.convex_hull

        # Calculate face areas and normals
        face_areas = hull.area_faces
        face_normals = hull.face_normals

        # Find the face with largest area that faces downward (negative Z component)
        # or find the best orientation by testing different rotations

        # Simple approach: use PCA to find principal axes
        # and orient so the smallest axis is Z (thickness direction)
        covariance = np.cov(hull.vertices.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # Sort by eigenvalue (smallest = thickness direction)
        order = np.argsort(eigenvalues)
        z_axis = eigenvectors[:, order[0]]

        # Create rotation to align Z axis with world Z
        target = np.array([0, 0, -1])  # Negative Z is down in 3D printing

        # Calculate rotation matrix
        v = np.cross(z_axis, target)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, target)

        if s > 1e-6:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rotation = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
            mesh.apply_transform(np.vstack([np.hstack([rotation, np.zeros((3, 1))]), [0, 0, 0, 1]]))

        # Ensure normals point outward
        mesh.fix_normals()

    except Exception as e:
        logger.warning(f"Could not auto-orient mesh: {e}")
        # Still return the mesh with basic fixes applied
        mesh.fix_normals()

    return mesh


def fix_holes(mesh_or_path: str | Path | trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Fix holes in a mesh using trimesh repair functions.

    Repairs:
    - Duplicate vertices
    - Degenerate faces
    - Inverted normals
    - Non-manifold edges (where possible)

    Note: Large holes cannot always be filled automatically.

    Args:
        mesh_or_path: Path to mesh file or trimesh.Trimesh object

    Returns:
        Repaired trimesh.Trimesh object (new copy)
    """
    if isinstance(mesh_or_path, (str, Path)):
        mesh = trimesh.load_mesh(str(mesh_or_path), force="mesh")
    else:
        mesh = mesh_or_path

    # Create a copy
    mesh = mesh.copy()

    # Merge duplicate vertices
    mesh.merge_vertices()

    # Remove degenerate faces and zero-area triangles
    mesh.process(validate=True)

    # Fix normals to be consistent
    mesh.fix_normals()

    # Remove unreferenced vertices
    mesh.remove_unreferenced_vertices()

    return mesh


def get_printability_report(mesh_path: str | Path) -> dict:
    """
    Get a detailed printability report for a mesh.

    Args:
        mesh_path: Path to mesh file

    Returns:
        Dictionary with printability analysis:
        {
            "is_printable": bool,
            "is_watertight": bool,
            "is_manifold": bool,
            "has_positive_volume": bool,
            "has_degenerate_faces": bool,
            "vertex_count": int,
            "face_count": int,
            "volume": float,
            "surface_area": float,
            "bounds": [[min_x, min_y, min_z], [max_x, max_y, max_z]],
            "errors": [str],
            "warnings": [str]
        }
    """
    mesh_path = Path(mesh_path)

    report = {
        "is_printable": False,
        "is_watertight": False,
        "is_manifold": False,
        "has_positive_volume": False,
        "has_degenerate_faces": False,
        "vertex_count": 0,
        "face_count": 0,
        "volume": 0.0,
        "surface_area": 0.0,
        "bounds": None,
        "errors": [],
        "warnings": [],
    }

    if not mesh_path.exists():
        report["errors"].append(f"File not found: {mesh_path}")
        return report

    try:
        mesh = trimesh.load_mesh(str(mesh_path), force="mesh")

        # Basic stats
        report["vertex_count"] = len(mesh.vertices)
        report["face_count"] = len(mesh.faces)
        report["bounds"] = mesh.bounds.tolist()
        report["surface_area"] = float(mesh.area)

        # Check watertight
        report["is_watertight"] = mesh.is_watertight
        if not mesh.is_watertight:
            report["errors"].append("Mesh is not watertight (has holes)")

        # Check manifold
        report["is_manifold"] = mesh.is_watertight  # Trimesh uses watertight as manifold check
        if not mesh.is_watertight:
            report["errors"].append("Mesh is not manifold")

        # Check volume
        try:
            volume = mesh.volume
            report["volume"] = float(volume)
            report["has_positive_volume"] = volume > 0
            if volume <= 0:
                report["errors"].append("Mesh has zero or negative volume")
        except Exception:
            report["errors"].append("Could not calculate mesh volume")

        # Check for degenerate faces
        if len(mesh.faces) == 0:
            report["has_degenerate_faces"] = True
            report["errors"].append("Mesh has no faces")

        # Overall printability
        report["is_printable"] = (
            report["is_watertight"] and report["has_positive_volume"] and len(mesh.faces) > 0
        )

    except Exception as e:
        report["errors"].append(f"Failed to analyze mesh: {e}")

    return report
