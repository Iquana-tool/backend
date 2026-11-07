"""Remember whether an object participates in automatic hierarchy assignment."""
from alembic import op
import sqlalchemy as sa

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("contours", sa.Column("hierarchy_enabled", sa.Boolean(),
                                      nullable=False, server_default=sa.true()))


def downgrade():
    op.drop_column("contours", "hierarchy_enabled")
