from typing import Optional, List, Generic, TypeVar, Type, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.exc import IntegrityError
from pydantic import BaseModel
import uuid

from app.models.models import Base

ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class BaseService(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Base service class with common CRUD operations"""
    
    def __init__(self, model: Type[ModelType]):
        self.model = model
    
    async def get_by_id(
        self,
        db: AsyncSession,
        id: uuid.UUID,
        options: Optional[List] = None
    ) -> Optional[ModelType]:
        """Get a record by ID"""
        query = select(self.model).where(self.model.id == id)
        
        if options:
            query = query.options(*options)
        
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_multi(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        options: Optional[List] = None,
        filters: Optional[Dict[str, Any]] = None,
        order_by = None
    ) -> List[ModelType]:
        """Get multiple records with optional filtering and ordering"""
        query = select(self.model)
        
        if options:
            query = query.options(*options)
        
        if filters:
            for key, value in filters.items():
                if hasattr(self.model, key):
                    query = query.where(getattr(self.model, key) == value)
        
        if order_by:
            query = query.order_by(order_by)
        else:
            query = query.order_by(self.model.created_at.desc())
        
        query = query.offset(skip).limit(limit)
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def create(
        self,
        db: AsyncSession,
        obj_in: CreateSchemaType,
        **kwargs
    ) -> ModelType:
        """Create a new record"""
        obj_data = obj_in.model_dump() if hasattr(obj_in, 'model_dump') else obj_in.dict()
        obj_data.update(kwargs)
        
        db_obj = self.model(**obj_data)
        db.add(db_obj)
        
        try:
            await db.commit()
            await db.refresh(db_obj)
            return db_obj
        except IntegrityError as e:
            await db.rollback()
            raise ValueError(f"Database integrity error: {str(e)}")
    
    async def update(
        self,
        db: AsyncSession,
        id: uuid.UUID,
        obj_in: UpdateSchemaType,
        **kwargs
    ) -> Optional[ModelType]:
        """Update a record"""
        # Get existing record
        db_obj = await self.get_by_id(db, id)
        if not db_obj:
            return None
        
        # Prepare update data
        update_data = obj_in.model_dump(exclude_unset=True) if hasattr(obj_in, 'model_dump') else obj_in.dict(exclude_unset=True)
        update_data.update(kwargs)
        
        # Update fields
        for field, value in update_data.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        
        try:
            await db.commit()
            await db.refresh(db_obj)
            return db_obj
        except IntegrityError as e:
            await db.rollback()
            raise ValueError(f"Database integrity error: {str(e)}")
    
    async def delete(
        self,
        db: AsyncSession,
        id: uuid.UUID
    ) -> bool:
        """Delete a record"""
        db_obj = await self.get_by_id(db, id)
        if not db_obj:
            return False
        
        await db.delete(db_obj)
        await db.commit()
        return True
    
    async def count(
        self,
        db: AsyncSession,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Count records with optional filtering"""
        query = select(func.count(self.model.id))
        
        if filters:
            for key, value in filters.items():
                if hasattr(self.model, key):
                    query = query.where(getattr(self.model, key) == value)
        
        result = await db.execute(query)
        return result.scalar()
    
    async def exists(
        self,
        db: AsyncSession,
        **filters
    ) -> bool:
        """Check if a record exists with given filters"""
        query = select(self.model)
        
        for key, value in filters.items():
            if hasattr(self.model, key):
                query = query.where(getattr(self.model, key) == value)
        
        query = query.limit(1)
        result = await db.execute(query)
        return result.scalar_one_or_none() is not None


class CRUDMixin:
    """Mixin class for common CRUD operations"""
    
    @classmethod
    async def create_record(
        cls,
        db: AsyncSession,
        **kwargs
    ):
        """Create a record using the model directly"""
        db_obj = cls(**kwargs)
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj
    
    @classmethod
    async def get_by_field(
        cls,
        db: AsyncSession,
        field_name: str,
        field_value: Any,
        options: Optional[List] = None
    ):
        """Get record by any field"""
        query = select(cls).where(getattr(cls, field_name) == field_value)
        
        if options:
            query = query.options(*options)
        
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    @classmethod
    async def get_multi_by_field(
        cls,
        db: AsyncSession,
        field_name: str,
        field_value: Any,
        options: Optional[List] = None,
        limit: int = None
    ):
        """Get multiple records by field"""
        query = select(cls).where(getattr(cls, field_name) == field_value)
        
        if options:
            query = query.options(*options)
        
        if limit:
            query = query.limit(limit)
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def update_fields(
        self,
        db: AsyncSession,
        **kwargs
    ):
        """Update specific fields of the instance"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        await db.commit()
        await db.refresh(self)
        return self


# Add the mixin to all models
Base.registry._class_registry.setdefault('CRUDMixin', CRUDMixin)

# Utility functions for complex queries
async def search_records(
    db: AsyncSession,
    model: Type[ModelType],
    search_term: str,
    search_fields: List[str],
    skip: int = 0,
    limit: int = 100,
    options: Optional[List] = None
) -> List[ModelType]:
    """Search records across multiple text fields"""
    query = select(model)
    
    if options:
        query = query.options(*options)
    
    # Build search conditions
    search_conditions = []
    for field in search_fields:
        if hasattr(model, field):
            search_conditions.append(
                getattr(model, field).ilike(f"%{search_term}%")
            )
    
    if search_conditions:
        query = query.where(or_(*search_conditions))
    
    query = query.offset(skip).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()


async def get_or_create(
    db: AsyncSession,
    model: Type[ModelType],
    defaults: Optional[Dict[str, Any]] = None,
    **kwargs
) -> tuple[ModelType, bool]:
    """Get existing record or create new one"""
    # Try to get existing record
    query = select(model)
    for key, value in kwargs.items():
        if hasattr(model, key):
            query = query.where(getattr(model, key) == value)
    
    result = await db.execute(query)
    instance = result.scalar_one_or_none()
    
    if instance:
        return instance, False
    
    # Create new record
    create_data = kwargs.copy()
    if defaults:
        create_data.update(defaults)
    
    instance = model(**create_data)
    db.add(instance)
    
    try:
        await db.commit()
        await db.refresh(instance)
        return instance, True
    except IntegrityError:
        await db.rollback()
        # Race condition - try to get again
        result = await db.execute(query)
        instance = result.scalar_one_or_none()
        if instance:
            return instance, False
        raise