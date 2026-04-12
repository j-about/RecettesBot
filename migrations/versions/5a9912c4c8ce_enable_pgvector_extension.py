"""enable pgvector extension

Revision ID: 5a9912c4c8ce
Revises:
Create Date: 2026-04-09 23:23:01.103606

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5a9912c4c8ce"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Enable the pgvector extension on the target database."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    """Drop the pgvector extension. Cascades to any vector columns."""
    op.execute("DROP EXTENSION IF EXISTS vector")
