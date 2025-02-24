import flask


def create_app():
    """Flask application factory function."""
    app = flask.Flask(__name__)

    # Import and register blueprints
    from app.routes.segmentation import segmentation_bp
    app.register_blueprint(segmentation_bp, url_prefix="/api")

    return app
