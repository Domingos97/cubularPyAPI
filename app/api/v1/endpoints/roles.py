from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user, get_current_admin_user, get_db
from app.models.models import User
from app.models.schemas import (
    RoleCreate, 
    RoleUpdate, 
    RoleResponse, 
    RoleStatistics,
    UserRoleAssignment,
    CheckUserRoleResponse,
    UserResponse
)
from app.services.role_service import RoleService

router = APIRouter()

@router.get("/", response_model=List[RoleResponse], tags=["roles"])
async def get_all_roles(
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all roles (admin only)
    """
    try:
        roles = await RoleService.get_all_roles(db)
        return roles
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch roles: {str(e)}"
        )

@router.get("/{role_id}", response_model=RoleResponse, tags=["roles"])
async def get_role_by_id(
    role_id: str,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get role by ID (admin only)
    """
    try:
        role = await RoleService.get_role_by_id(db, role_id)
        if not role:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Role not found"
            )
        return role
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch role: {str(e)}"
        )

@router.post("/", response_model=RoleResponse, status_code=status.HTTP_201_CREATED, tags=["roles"])
async def create_role(
    role_data: RoleCreate,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new role (admin only)
    """
    try:
        role = await RoleService.create_role(db, role_data)
        return role
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create role: {str(e)}"
        )

@router.put("/{role_id}", response_model=RoleResponse, tags=["roles"])
async def update_role(
    role_id: str,
    role_data: RoleUpdate,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update a role (admin only)
    """
    try:
        role = await RoleService.update_role(db, role_id, role_data)
        if not role:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Role not found"
            )
        return role
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update role: {str(e)}"
        )

@router.delete("/{role_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["roles"])
async def delete_role(
    role_id: str,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a role (admin only)
    """
    try:
        success = await RoleService.delete_role(db, role_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Role not found"
            )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete role: {str(e)}"
        )

@router.get("/{role_id}/users", response_model=List[UserResponse], tags=["roles"])
async def get_users_by_role_id(
    role_id: str,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all users with a specific role by role ID (admin only)
    """
    try:
        users = await RoleService.get_users_by_role_id(db, role_id)
        return users
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch users by role: {str(e)}"
        )

@router.get("/name/{role_name}/users", response_model=List[UserResponse], tags=["roles"])
async def get_users_by_role_name(
    role_name: str,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all users with a specific role by role name (admin only)
    """
    try:
        users = await RoleService.get_users_by_role_name(db, role_name)
        return users
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch users by role: {str(e)}"
        )

@router.get("/check/{user_id}/{role_name}", response_model=CheckUserRoleResponse, tags=["roles"])
async def check_user_role(
    user_id: str,
    role_name: str,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Check if a user has a specific role (admin only)
    """
    try:
        has_role = await RoleService.user_has_role(db, user_id, role_name)
        return CheckUserRoleResponse(has_role=has_role)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check user role: {str(e)}"
        )

@router.get("/statistics", response_model=RoleStatistics, tags=["roles"])
async def get_role_statistics(
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get role usage statistics (admin only)
    """
    try:
        stats = await RoleService.get_role_statistics(db)
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch role statistics: {str(e)}"
        )

@router.post("/assign", response_model=UserResponse, tags=["roles"])
async def assign_role_to_user(
    assignment_data: UserRoleAssignment,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Assign a role to a user (admin only)
    """
    try:
        user = await RoleService.assign_role_to_user(
            db, 
            str(assignment_data.user_id), 
            str(assignment_data.roleid)
        )
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to assign role: {str(e)}"
        )