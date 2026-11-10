"""
Capa 03: Núcleo Espacial XR, Físicas 3D y Persistencia de Proyectos
"""
from .spatial_hologram_3d import SpatialHologram3D, HandVectorState
from .nexus_physics_3d import SpatialPhysicsSolver3D, RigidBodyHologram3D, HandColliderNode
from .nexus_workspace_project import NXSProjectSerializer, SerializableHologram3D

__all__ = [
    "SpatialHologram3D",
    "HandVectorState",
    "SpatialPhysicsSolver3D",
    "RigidBodyHologram3D",
    "HandColliderNode",
    "NXSProjectSerializer",
    "SerializableHologram3D",
]
