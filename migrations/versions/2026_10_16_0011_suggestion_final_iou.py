"""AI suggestion edit size: how far the final outline moved from the suggestion.

New nullable column ``ai_suggestions.final_iou``, written on every hand edit of an
object. Rows that were edited before this revision stay NULL here; filling them
needs the outline comparison in application code, which a migration must not
import, and on a large log it would also hold up the boot-time upgrade. Run
``scripts/backfill_suggestion_iou.py`` once after upgrading instead.

Revision ID: 0011
Revises: 0010
Create Date: 2026-10-16 10:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0011'
down_revision: Union[str, Sequence[str], None] = '0010'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('ai_suggestions', sa.Column('final_iou', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('ai_suggestions', 'final_iou')
