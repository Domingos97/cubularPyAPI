"""add has_ai_personalities_access column to users

Revision ID: add_ai_personalities_access
Revises: 
Create Date: 2025-10-19 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_ai_personalities_access'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Add column to users table
    op.add_column('users', sa.Column('has_ai_personalities_access', sa.Boolean(), server_default=sa.false(), nullable=False))


def downgrade():
    # Remove column from users table
    op.drop_column('users', 'has_ai_personalities_access')
