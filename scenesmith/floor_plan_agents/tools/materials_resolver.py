"""Materials resolver for floor plan generation.

Abstracts material selection from either local defaults or materials retrieval server.
"""

import logging

from dataclasses import dataclass
from pathlib import Path

from scenesmith.agent_utils.materials_retrieval_server.client import (
    MaterialsRetrievalClient,
)
from scenesmith.agent_utils.materials_retrieval_server.dataclasses import (
    MaterialsRetrievalServerRequest,
)
from scenesmith.utils.material import Material

console_logger = logging.getLogger(__name__)


# Default materials directory at repository root.
DEFAULT_MATERIALS_DIR = Path(__file__).parent.parent.parent.parent / "materials"


@dataclass
class MaterialsConfig:
    """Configuration for materials resolution."""

    use_retrieval_server: bool = False
    """If True, use materials retrieval server with CLIP semantic search."""

    default_wall_material: str = "Plaster001_1K-JPG"
    """Default wall material name."""

    default_floor_material: str = "Wood094_1K-JPG"
    """Default floor material name."""

    materials_dir: Path | None = None
    """Override materials directory (default: materials/ at repo root)."""

    output_dir: Path | None = None
    """Output directory for scene. Materials from server will be copied here."""


class MaterialsResolver:
    """Resolves material requests to paths.

    Supports two modes:
    1. Local: Returns default wall/floor materials (only 2 local materials exist)
    2. Server: CLIP semantic search via materials retrieval server
    """

    def __init__(self, config: MaterialsConfig | None = None):
        """Initialize materials resolver.

        Args:
            config: Materials configuration. If None, uses defaults.
        """
        self.config = config or MaterialsConfig()
        self.materials_dir = self.config.materials_dir or DEFAULT_MATERIALS_DIR
        self.output_dir = self.config.output_dir

        # Cache discovered materials from local directory.
        self._available_materials: dict[str, Path] | None = None

        # Cache materials retrieved from server (material_id -> material_path).
        self._server_materials_cache: dict[str, Path] = {}

    def get_material(self, description: str) -> Material | None:
        """Get material matching description.

        Args:
            description: Description of desired material
                (e.g., "warm oak hardwood", "white painted drywall").

        Returns:
            Material if found, None otherwise.
        """
        if self.config.use_retrieval_server:
            return self._get_from_server(description)

        # Local mode: only 2 materials exist, pick based on simple check.
        desc_lower = description.lower()
        if any(w in desc_lower for w in ["floor", "wood", "tile", "carpet", "parquet"]):
            return self.get_default_floor_material()
        return self.get_default_wall_material()

    def get_default_wall_material(self) -> Material | None:
        """Get default wall material."""
        return self._resolve_material_id(self.config.default_wall_material)

    def get_default_floor_material(self) -> Material | None:
        """Get default floor material."""
        return self._resolve_material_id(self.config.default_floor_material)

    def get_material_by_id(self, material_id: str) -> Material | None:
        """Get material by exact ID.

        Args:
            material_id: Material identifier (e.g., 'Plaster001_1K-JPG').

        Returns:
            Material if found, None otherwise.
        """
        return self._resolve_material_id(material_id)

    def _resolve_material_id(self, material_id: str) -> Material | None:
        """Resolve a material ID to Material.

        Args:
            material_id: Material identifier.

        Returns:
            Material if found, None otherwise.
        """
        normalized_id = material_id.strip()
        if not normalized_id:
            return None

        # First check server cache.
        if normalized_id in self._server_materials_cache:
            return Material(
                path=self._server_materials_cache[normalized_id],
                material_id=normalized_id,
            )

        # Then check local materials directory.
        self._discover_materials()

        if normalized_id in self._available_materials:
            # Local materials use folder name as ID.
            return Material.from_path(self._available_materials[normalized_id])

        # Case-insensitive fallback.
        lower_to_name = {
            name.lower(): name for name in self._available_materials.keys()
        }
        matched_name = lower_to_name.get(normalized_id.lower())
        if matched_name:
            return Material.from_path(self._available_materials[matched_name])

        # Robust fallback for default IDs when exact match is unavailable.
        fallback = self._fallback_default_material(normalized_id)
        if fallback is not None:
            console_logger.warning(
                f"Material '{normalized_id}' not found; using fallback "
                f"'{fallback.material_id}'."
            )
            return fallback

        console_logger.warning(f"Material not found: {normalized_id}")
        return None

    def _fallback_default_material(self, material_id: str) -> Material | None:
        """Best-effort fallback for missing default material IDs."""
        if not self._available_materials:
            return None

        requested = material_id.lower()
        names = list(self._available_materials.keys())

        # Prefer semantically similar material folders.
        preferred_keywords: list[str] = []
        if "wood" in requested or "floor" in requested:
            preferred_keywords = ["wood", "parquet", "floor", "tile"]
        elif "plaster" in requested or "wall" in requested or "stucco" in requested:
            preferred_keywords = ["plaster", "stucco", "paint", "wall", "brick"]

        for keyword in preferred_keywords:
            for name in names:
                if keyword in name.lower():
                    return Material.from_path(self._available_materials[name])

        # Last resort: pick first available material for resilience.
        return Material.from_path(self._available_materials[names[0]])

    def _discover_materials(self) -> None:
        """Discover available materials in materials directory."""
        if self._available_materials is not None:
            return

        self._available_materials = {}

        if not self.materials_dir.exists():
            console_logger.warning(
                f"Materials directory not found: {self.materials_dir}"
            )
            return

        # Find all material directories (contain texture files).
        for path in self.materials_dir.iterdir():
            if path.is_dir():
                # Check if it contains texture files.
                has_textures = any(
                    f.suffix.lower() in {".jpg", ".jpeg", ".png"}
                    for f in path.iterdir()
                    if f.is_file()
                )
                if has_textures:
                    self._available_materials[path.name] = path

        console_logger.debug(
            f"Discovered {len(self._available_materials)} materials in "
            f"{self.materials_dir}"
        )

    def _get_from_server(self, description: str) -> Material | None:
        """Get material from retrieval server using CLIP semantic search.

        Args:
            description: Material description.

        Returns:
            Material if found, None otherwise.
        """
        # Use scene output directory for self-contained output.
        if not self.output_dir:
            raise ValueError(
                "output_dir must be set in MaterialsConfig when "
                "use_retrieval_server=True. Materials need a persistent location."
            )

        materials_output_dir = self.output_dir / "materials"
        materials_output_dir.mkdir(parents=True, exist_ok=True)

        client = MaterialsRetrievalClient()
        request = MaterialsRetrievalServerRequest(
            material_description=description,
            output_dir=str(materials_output_dir),
            num_candidates=1,
        )

        for _index, response in client.retrieve_materials([request]):
            if response.results:
                result = response.results[0]
                material_path = Path(result.material_path)

                # Cache the material so get_material_by_id can find it later.
                self._server_materials_cache[result.material_id] = material_path

                return Material(path=material_path, material_id=result.material_id)

        return None
