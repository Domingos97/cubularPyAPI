import os
from alembic.config import Config
from alembic import command


def generate_single_migration():
    """
    Generate a single Alembic migration file reflecting the current state of models vs DB.
    """
    # Path to alembic.ini (assumed to be in project root)
    alembic_ini_path = os.path.join(os.path.dirname(__file__), 'alembic.ini')
    if not os.path.exists(alembic_ini_path):
        raise FileNotFoundError("alembic.ini not found. Please ensure Alembic is set up.")

    alembic_cfg = Config(alembic_ini_path)
    # Generate a new migration file with autogenerate
    command.revision(alembic_cfg, message="autogenerate current schema", autogenerate=True)

if __name__ == "__main__":
    generate_single_migration()
