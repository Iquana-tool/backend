"""Merge the three migration lines that branched from 0011.

The active-learning queue (0012-0015), account invites (0016) and contour
hierarchy choice (0017) were developed in parallel on top of 0011. This
revision joins them into a single head so `alembic upgrade head` works again.
It changes no schema.

Revision ID: 0018
Revises: 0015, 0016, 0017
Create Date: 2026-11-19
"""
from typing import Sequence, Union

revision: str = "0018"
down_revision: Union[str, Sequence[str], None] = ("0015", "0016", "0017")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
