from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, and_, func, delete
import uuid

from app.models.models import Role, User
from app.models.schemas import RoleCreate, RoleUpdate
from app.services.base import BaseService


class RoleService(BaseService):
    """Role management service"""
    
    @staticmethod
    async def get_all_roles(db: AsyncSession) -> List[Role]:
        """Get all roles"""
        query = select(Role).order_by(Role.role)
        result = await db.execute(query)
        return result.scalars().all()
    
    @staticmethod
    async def get_role_by_id(db: AsyncSession, role_id: str) -> Optional[Role]:
        """Get role by ID"""
        try:
            role_uuid = uuid.UUID(role_id)
            query = select(Role).where(Role.id == role_uuid)
            result = await db.execute(query)
            return result.scalar_one_or_none()
        except ValueError:
            return None
    
    @staticmethod
    async def get_role_by_name(db: AsyncSession, role_name: str) -> Optional[Role]:
        """Get role by name"""
        query = select(Role).where(Role.role == role_name)
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def create_role(db: AsyncSession, role_data: RoleCreate) -> Role:
        """Create a new role"""
        # Check if role already exists
        existing_role = await RoleService.get_role_by_name(db, role_data.role)
        if existing_role:
            raise ValueError(f"Role '{role_data.role}' already exists")
        
        # Create new role
        db_role = Role(
            role=role_data.role
        )
        
        db.add(db_role)
        await db.commit()
        await db.refresh(db_role)
        
        return db_role
    
    @staticmethod
    async def update_role(
        db: AsyncSession, 
        role_id: str, 
        role_data: RoleUpdate
    ) -> Optional[Role]:
        """Update an existing role"""
        role = await RoleService.get_role_by_id(db, role_id)
        if not role:
            return None
        
        # Check if new role name already exists (if changing name)
        if role_data.role and role_data.role != role.role:
            existing_role = await RoleService.get_role_by_name(db, role_data.role)
            if existing_role:
                raise ValueError(f"Role '{role_data.role}' already exists")
        
        # Update role
        if role_data.role is not None:
            role.role = role_data.role
        
        await db.commit()
        await db.refresh(role)
        
        return role
    
    @staticmethod
    async def delete_role(db: AsyncSession, role_id: str) -> bool:
        """Delete a role"""
        role = await RoleService.get_role_by_id(db, role_id)
        if not role:
            return False
        
        # Check if any users have this role
        users_with_role = await RoleService.get_users_by_role_id(db, role_id)
        if users_with_role:
            raise ValueError(f"Cannot delete role '{role.role}' because it is assigned to {len(users_with_role)} users")
        
        # Delete the role
        await db.delete(role)
        await db.commit()
        
        return True
    
    @staticmethod
    async def get_users_by_role_id(db: AsyncSession, role_id: str) -> List[User]:
        """Get all users with a specific role by role ID"""
        try:
            role_uuid = uuid.UUID(role_id)
            query = (
                select(User)
                .options(selectinload(User.role))
                .where(User.roleid == role_uuid)
                .order_by(User.username)
            )
            result = await db.execute(query)
            return result.scalars().all()
        except ValueError:
            return []
    
    @staticmethod
    async def get_users_by_role_name(db: AsyncSession, role_name: str) -> List[User]:
        """Get all users with a specific role by role name"""
        query = (
            select(User)
            .options(selectinload(User.role))
            .join(Role, User.roleid == Role.id)
            .where(Role.role == role_name)
            .order_by(User.username)
        )
        result = await db.execute(query)
        return result.scalars().all()
    
    @staticmethod
    async def user_has_role(db: AsyncSession, user_id: str, role_name: str) -> bool:
        """Check if a user has a specific role"""
        try:
            user_uuid = uuid.UUID(user_id)
            query = (
                select(User)
                .options(selectinload(User.role))
                .where(User.id == user_uuid)
            )
            result = await db.execute(query)
            user = result.scalar_one_or_none()
            
            if not user or not user.role:
                return False
            
            return user.role.role == role_name
        except ValueError:
            return False
    
    @staticmethod
    async def get_role_statistics(db: AsyncSession) -> Dict[str, Any]:
        """Get role usage statistics"""
        # Get total number of roles
        total_roles_query = select(func.count(Role.id))
        total_roles_result = await db.execute(total_roles_query)
        total_roles = total_roles_result.scalar()
        
        # Get role usage statistics
        role_usage_query = (
            select(
                Role.role,
                func.count(User.id).label('user_count')
            )
            .outerjoin(User, User.roleid == Role.id)
            .group_by(Role.id, Role.role)
            .order_by(Role.role)
        )
        
        result = await db.execute(role_usage_query)
        role_usage = [
            {"role": row.role, "user_count": row.user_count}
            for row in result.fetchall()
        ]
        
        # Get total number of users
        total_users_query = select(func.count(User.id))
        total_users_result = await db.execute(total_users_query)
        total_users = total_users_result.scalar()
        
        return {
            "total_roles": total_roles,
            "total_users": total_users,
            "role_usage": role_usage
        }
    
    @staticmethod
    async def assign_role_to_user(
        db: AsyncSession, 
        user_id: str, 
        role_id: str
    ) -> Optional[User]:
        """Assign a role to a user"""
        # Get user
        try:
            user_uuid = uuid.UUID(user_id)
            user_query = select(User).where(User.id == user_uuid)
            user_result = await db.execute(user_query)
            user = user_result.scalar_one_or_none()
            
            if not user:
                return None
            
            # Get role
            role = await RoleService.get_role_by_id(db, role_id)
            if not role:
                raise ValueError(f"Role with ID '{role_id}' not found")
            
            # Assign role
            user.roleid = role.id
            await db.commit()
            await db.refresh(user, ["role"])
            
            return user
        except ValueError:
            return None