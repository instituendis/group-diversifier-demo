from __future__ import annotations
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from .dimension_config import (
        Profile,
        ValidationResult
    )
except ImportError:
    from dimension_config import (  # type: ignore[no-redef]
        Profile,
        ValidationResult
    )

try:
    from app import hosting as _hosting
except ImportError:  # pragma: no cover - imported without the project root on sys.path
    _hosting = None  # type: ignore[assignment]

try:
    from app import extensions as _extensions
except ImportError:  # pragma: no cover - imported without the project root on sys.path
    _extensions = None  # type: ignore[assignment]

PROFILE_SUFFIXES = (".yaml", ".yml")

DEFAULT_PROFILE = "default"
DEFAULT_ALIAS = "demo"


def _extension_hook(name: str, default: Any = None) -> Any:
    if _extensions is None:
        return default
    return _extensions.hook(name, default)


def _extension_installed() -> bool:
    return _extensions is not None and _extensions.installed()


def _is_bare_name(name: str) -> bool:
    if not name or name in (".", "..") or "\x00" in name:
        return False
    if "/" in name or "\\" in name or Path(name).is_absolute():
        return False
    return Path(name).name == name


class ProfilesReadOnly(ValueError): ...


def profiles_read_only() -> bool:
    return bool(_hosting is not None and _hosting.hosted())


class ConfigurationLoader:
    def __init__(self, config_dir: Optional[Path] = None):
        if config_dir is None:
            self.config_dir = Path(__file__).parent.parent.parent / "config"
        else:
            self.config_dir = Path(config_dir)
        
        # INVARIANT: never create the config directory in hosted mode: config/ may be
        # a read-only mount there, and a missing one just lists nothing.
        if not profiles_read_only():
            self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def load_profile(self, profile_name: str) -> Profile:
        profile_path = self.profile_path(profile_name)

        if profile_path is None:
            raise FileNotFoundError(
                f"Profile not found: {self.config_dir / (profile_name + '.yaml')}\n"
                f"Available profiles: {', '.join(self.load_all_profiles())}"
            )
        
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile_data = yaml.safe_load(f)
            
            if not profile_data:
                raise ValueError(
                    f"Profile file is empty: {profile_path}"
                )

            profile = Profile.from_dict(profile_data)

            if not profile.is_valid():
                raise ValueError(
                    f"Profile '{profile_name}' is invalid. "
                    f"Please check dimension configurations."
                )
            
            return profile
            
        except yaml.YAMLError as e:
            raise ValueError(
                f"Invalid YAML in profile '{profile_name}': {e}"
            ) from e
        except KeyError as e:
            raise ValueError(
                f"Profile '{profile_name}' is missing required field: {e}"
            ) from e
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(
                f"Error loading profile '{profile_name}': {e}"
            ) from e

    def profile_path(self, profile_name: str) -> Optional[Path]:
        name = str(profile_name)
        if not _is_bare_name(name):
            return None
        if name == DEFAULT_PROFILE:
            return self._default_path()
        return self._file_for(name)

    def _file_for(self, name: str) -> Optional[Path]:
        for suffix in PROFILE_SUFFIXES:
            candidate = self.config_dir / f"{name}{suffix}"
            if candidate.is_file():
                return candidate
        return None

    def _default_path(self) -> Optional[Path]:
        own = self._file_for(DEFAULT_PROFILE)
        if own is not None:
            return own
        if _extension_installed():
            rel = _extension_hook("default_profile_path")
            if not rel:
                return None
            candidate = self.config_dir / rel
            return candidate if candidate.is_file() else None
        return self._file_for(DEFAULT_ALIAS)

    def load_all_profiles(self) -> List[str]:
        profiles = []

        for profile_file in self.config_dir.glob("*.yaml"):
            if profile_file.is_file():
                profiles.append(profile_file.stem)

        for profile_file in self.config_dir.glob("*.yml"):
            if profile_file.is_file():
                profiles.append(profile_file.stem)

        if _extension_installed() and self._default_path() is not None:
            profiles.append(DEFAULT_PROFILE)

        profiles = sorted(list(set(profiles)))
        
        return profiles
    
    def save_profile(self, profile: Profile, profile_name: Optional[str] = None) -> Path:
        if profiles_read_only():
            raise ProfilesReadOnly(
                "Profiles cannot be saved on this hosted server: the profile files are "
                "shared by every user, so they are read-only here.")
        if not profile.is_valid():
            raise ValueError(
                "Cannot save invalid profile. "
                "Please check dimension configurations."
            )

        if profile_name is None:
            profile_name = profile.name

        if not profile_name:
            raise ValueError("Profile name cannot be empty")

        profile_name = "".join(
            c for c in profile_name 
            if c.isalnum() or c in ('-', '_')
        ).lower()

        profile_path = self.config_dir / f"{profile_name}.yaml"
        
        try:
            with open(profile_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    profile.to_dict(),
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True
                )
            
            return profile_path
            
        except Exception as e:
            raise ValueError(
                f"Error saving profile to {profile_path}: {e}"
            )
    
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)

        required_fields = ["name", "description", "version", "dimensions"]
        for field in required_fields:
            if field not in config:
                result.add_error(f"Missing required field: {field}")

        if "dimensions" in config:
            for idx, dim_config in enumerate(config["dimensions"]):
                self._validate_dimension(dim_config, idx, result)

        self._validate_weights(config, result)
        
        return result
    
    def _validate_dimension(
        self,
        dim_config: Dict[str, Any],
        idx: int,
        result: ValidationResult
    ) -> None:
        required_fields = ["name", "type"]
        for field in required_fields:
            if field not in dim_config:
                result.add_error(
                    f"Dimension {idx}: Missing required field '{field}'"
                )

        valid_types = ["categorical", "ordinal", "multi_select", "numeric"]
        if "type" in dim_config and dim_config["type"] not in valid_types:
            result.add_error(
                f"Dimension {idx}: Invalid type '{dim_config['type']}'. "
                f"Valid types: {', '.join(valid_types)}"
            )

        dim_type = dim_config.get("type")
        
        if dim_type == "categorical":
            if "categories" not in dim_config:
                result.add_error(
                    f"Dimension {idx}: Categorical dimensions require 'categories' field"
                )

        elif dim_type == "ordinal":
            if "levels" not in dim_config or not dim_config["levels"]:
                result.add_error(
                    f"Dimension {idx}: Ordinal dimensions require 'levels' field"
                )
        
        elif dim_type == "multi_select":
            if "values" not in dim_config or not dim_config["values"]:
                result.add_error(
                    f"Dimension {idx}: Multi-select dimensions require 'values' field"
                )
            if "max_selections" not in dim_config:
                result.add_warning(
                    f"Dimension {idx}: Multi-select dimensions should have 'max_selections'"
                )
        
        elif dim_type == "numeric":
            if "min_value" not in dim_config or "max_value" not in dim_config:
                result.add_error(
                    f"Dimension {idx}: Numeric dimensions require 'min_value' and 'max_value'"
                )
            if dim_config.get("min_value") is not None and dim_config.get("max_value") is not None:
                if dim_config["min_value"] >= dim_config["max_value"]:
                    result.add_error(
                        f"Dimension {idx}: min_value must be less than max_value"
                    )
    
    def _validate_weights(
        self,
        config: Dict[str, Any],
        result: ValidationResult
    ) -> None:
        if "dimensions" not in config:
            return

        for idx, dim_config in enumerate(config["dimensions"]):
            weight = dim_config.get("weight", 50.0)
            if weight < 0:
                result.add_error(
                    f"Dimension {idx}: Weight cannot be negative (found: {weight})"
                )
            elif weight > 200:
                result.add_warning(
                    f"Dimension {idx}: Weight is very high ({weight}). "
                    "Consider using smaller weights (recommended: 0-100)."
                )

        for idx, dim_config in enumerate(config["dimensions"]):
            if dim_config.get("enabled", True) and dim_config.get("weight", 50.0) == 0:
                result.add_warning(
                    f"Dimension {idx}: Dimension is enabled but has zero weight. "
                    "This dimension will not affect team diversity."
                )
    

    def create_default_profile(self) -> Profile:
        factory = _extension_hook("default_profile")
        if factory is not None:
            return factory()
        try:
            return self.load_profile(DEFAULT_ALIAS)
        except (FileNotFoundError, ValueError):
            return Profile(
                name="Default",
                description="No dimensions defined",
                version="1.0",
                dimensions=[],
            )
