"""Change column type

Revision ID: e8c18df3bb7e
Revises: 7eae3020adcb
Create Date: 2025-03-25 12:51:01.050748

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e8c18df3bb7e'
down_revision: Union[str, None] = '7eae3020adcb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('game_lines', 'spread_outcome',
               existing_type=sa.INTEGER(),
               nullable=True)
    op.execute("ALTER TABLE game_lines ALTER COLUMN money_outcome TYPE INTEGER USING money_outcome::integer;")

    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.execute("ALTER TABLE game_lines ALTER COLUMN money_outcome TYPE TEXT USING money_outcome::text;")  # Adjust to previous type

    op.alter_column('game_lines', 'spread_outcome',
               existing_type=sa.INTEGER(),
               nullable=False)
    # ### end Alembic commands ###
