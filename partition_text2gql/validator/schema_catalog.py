"""
Schema Catalog representation for graph domains.
Stores allowed node labels, allowed relationship types, and allowed properties per label.
"""

from typing import Dict, Set, List, Optional
from pydantic import BaseModel, Field


class LabelSchema(BaseModel):
    name: str
    properties: Set[str] = Field(default_factory=set)


class RelSchema(BaseModel):
    type_name: str
    source_labels: Set[str] = Field(default_factory=set)
    target_labels: Set[str] = Field(default_factory=set)
    properties: Set[str] = Field(default_factory=set)


class GraphDomainSchema(BaseModel):
    domain: str
    labels: Dict[str, LabelSchema] = Field(default_factory=dict)
    relationships: Dict[str, RelSchema] = Field(default_factory=dict)

    def add_label(self, name: str, properties: Optional[List[str]] = None) -> None:
        props = set(properties or [])
        if name in self.labels:
            self.labels[name].properties.update(props)
        else:
            self.labels[name] = LabelSchema(name=name, properties=props)

    def add_relationship(
        self,
        type_name: str,
        source: Optional[str] = None,
        target: Optional[str] = None,
        properties: Optional[List[str]] = None
    ) -> None:
        props = set(properties or [])
        if type_name in self.relationships:
            rel = self.relationships[type_name]
            if source:
                rel.source_labels.add(source)
            if target:
                rel.target_labels.add(target)
            rel.properties.update(props)
        else:
            sources = {source} if source else set()
            targets = {target} if target else set()
            self.relationships[type_name] = RelSchema(
                type_name=type_name,
                source_labels=sources,
                target_labels=targets,
                properties=props
            )

    def has_label(self, label: str) -> bool:
        return label in self.labels

    def has_relationship(self, rel_type: str) -> bool:
        return rel_type in self.relationships

    def has_property(self, label: Optional[str], prop: str) -> bool:
        if label and label in self.labels:
            return prop in self.labels[label].properties
        # Fallback: check across all labels in domain
        for l_schema in self.labels.values():
            if prop in l_schema.properties:
                return True
        return False

