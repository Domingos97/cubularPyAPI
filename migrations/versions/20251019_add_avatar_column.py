"""add avatar column to users

Revision ID: add_avatar_column
Revises: add_ai_personalities_access
Create Date: 2025-10-19 00:10:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_avatar_column'
down_revision = 'add_ai_personalities_access'
branch_labels = None
depends_on = None


def upgrade():
    # Add nullable avatar column to users table (stores path or URL)
    op.add_column('users', sa.Column('avatar', sa.String(length=255), nullable=True))


def downgrade():
    # Remove avatar column from users table
    op.drop_column('users', 'avatar')
