from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class DimensionType(Enum):
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    MULTI_SELECT = "multi_select"
    NUMERIC = "numeric"


@dataclass
class Level:
    name: str
    weight: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "weight": self.weight
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Level:
        return cls(
            name=data["name"],
            weight=data["weight"]
        )


@dataclass
class DimensionConfig:
    name: str
    type: str
    enabled: bool = True
    weight: float = 50.0

    categories: List[str] = field(default_factory=list)

    levels: List[Level] = field(default_factory=list)

    values: List[str] = field(default_factory=list)
    max_selections: Optional[int] = None

    min_value: Optional[float] = None
    max_value: Optional[float] = None

    default_category: Optional[str] = None
    default_value: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "enabled": self.enabled,
            "weight": self.weight,
            "categories": self.categories,
            "levels": [level.to_dict() for level in self.levels],
            "values": self.values,
            "max_selections": self.max_selections,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "default_category": self.default_category,
            "default_value": self.default_value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DimensionConfig:
        levels_data = data.get("levels", [])
        levels = [Level.from_dict(level) for level in levels_data] if levels_data else []
        
        return cls(
            name=data["name"],
            type=data["type"],
            enabled=data.get("enabled", True),
            weight=data.get("weight", 50.0),
            categories=data.get("categories", []),
            levels=levels,
            values=data.get("values", []),
            max_selections=data.get("max_selections"),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            default_category=data.get("default_category"),
            default_value=data.get("default_value")
        )
    
    def is_valid(self) -> bool:
        if self.type not in [dt.value for dt in DimensionType]:
            return False

        if self.type == "categorical":
            return True
        
        elif self.type == "ordinal":
            return len(self.levels) > 0
        
        elif self.type == "multi_select":
            return len(self.values) > 0 and self.max_selections is not None
        
        elif self.type == "numeric":
            return self.min_value is not None and self.max_value is not None
        
        return True
    
    def get_default(self) -> Any:
        if self.default_value is not None:
            return self.default_value
        
        if self.type == "categorical":
            return self.default_category or (self.categories[0] if self.categories else None)
        
        elif self.type == "ordinal":
            return self.levels[0].name if self.levels else None
        
        elif self.type == "numeric":
            return self.min_value
        
        elif self.type == "multi_select":
            return self.values[:self.max_selections] if self.values else []
        
        return None


@dataclass
class Profile:
    name: str
    description: str
    version: str
    dimensions: List[DimensionConfig]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "dimensions": [dim.to_dict() for dim in self.dimensions]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Profile:
        dimensions_data = data.get("dimensions", [])
        dimensions = [
            DimensionConfig.from_dict(dim) for dim in dimensions_data
        ] if dimensions_data else []
        
        return cls(
            name=data["name"],
            description=data["description"],
            version=data["version"],
            dimensions=dimensions
        )
    
    def get_dimension(self, name: str) -> Optional[DimensionConfig]:
        for dim in self.dimensions:
            if dim.name == name:
                return dim
        return None
    
    def get_enabled_dimensions(self) -> List[DimensionConfig]:
        return [dim for dim in self.dimensions if dim.enabled]
    
    def get_dimension_names(self) -> List[str]:
        return [dim.name for dim in self.dimensions]
    
    def is_valid(self) -> bool:
        if not self.name or not self.version:
            return False

        for dim in self.dimensions:
            if not dim.is_valid():
                return False

        names = self.get_dimension_names()
        if len(names) != len(set(names)):
            return False
        
        return True


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        self.warnings.append(message)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings
        }

