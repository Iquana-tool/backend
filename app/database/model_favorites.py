from sqlalchemy import Column, String, ForeignKey

from app.database import database


class UserModelFavorites(database):
    """A user's favorite (default) model per task.

    One row per ``(username, task)``: the model preselected for that task in the
    annotation page. Favorites are per-task on purpose -- a user may prefer one
    model for prompted segmentation and another for suggestion. A model that
    serves several tasks can be the favorite for each of them (one row per task).
    """

    __tablename__ = "user_model_favorites"

    username = Column(
        String,
        ForeignKey("users.username", ondelete="CASCADE"),
        primary_key=True,
    )
    # The task tag, e.g. "prompted-segmentation" / "instance-suggestion" /
    # "instance-segmentation". Matches the ai-service task surfaces.
    task = Column(String, primary_key=True)
    # Registry key of the favorited model (not a FK: models live in MLflow, not
    # this database).
    model_registry_key = Column(String, nullable=False)
