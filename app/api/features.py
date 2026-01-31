"""
Features API

Provides endpoints for listing available feature groups.
"""

from fastapi import APIRouter

from app.ml.features import feature_registry

router = APIRouter(prefix="/features", tags=["Features"])


@router.get("/groups")
def list_feature_groups():
    """
    List all available feature groups.
    
    Returns:
        List of feature groups with their descriptions and features.
    """
    return {
        "groups": feature_registry.get_group_info(),
        "total_groups": len(feature_registry.list_groups()),
    }


@router.get("/groups/{group_name}")
def get_feature_group(group_name: str):
    """
    Get details for a specific feature group.
    
    Args:
        group_name: Name of the feature group
    
    Returns:
        Feature group details or 404 if not found.
    """
    group = feature_registry.get(group_name)
    
    if not group:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Feature group not found: {group_name}")
    
    return {
        "name": group.name,
        "description": group.description,
        "features": group.feature_names,
        "dependencies": group.dependencies,
    }


@router.get("/all")
def list_all_features():
    """
    List all available features across all groups.
    
    Returns:
        List of all feature names.
    """
    all_features = feature_registry.get_feature_names(feature_registry.list_groups())
    
    return {
        "features": all_features,
        "total": len(all_features),
    }
