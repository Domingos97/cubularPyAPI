"""Add is_default field to ai_personalities table

Revision ID: add_is_default_to_personalities
Revises: 
Create Date: 2025-10-11

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_is_default_to_personalities'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Add is_default column to ai_personalities table
    op.add_column('ai_personalities', 
                  sa.Column('is_default', sa.Boolean(), default=False, nullable=False))
    
    # Set one personality as default if none exists
    connection = op.get_bind()
    result = connection.execute(
        sa.text("SELECT COUNT(*) FROM ai_personalities WHERE is_default = true")
    ).scalar()
    
    if result == 0:
        # Set the first active personality as default
        connection.execute(
            sa.text("""
                UPDATE ai_personalities 
                SET is_default = true 
                WHERE id = (
                    SELECT id FROM ai_personalities 
                    WHERE is_active = true 
                    ORDER BY created_at ASC 
                    LIMIT 1
                )
            """)
        )


def downgrade():
    # Remove is_default column from ai_personalities table
    op.drop_column('ai_personalities', 'is_default')