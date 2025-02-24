from marshmallow import Schema, fields, ValidationError, validates
from app.database.images import Images


class PointPromptSchema(Schema):
    """ Schema for validating a point annotation. """
    x = fields.Float(required=True)
    y = fields.Float(required=True)
    label = fields.Integer(required=True)

    @validates("label")
    def validate_label(self, value):
        if value not in [0, 1]:
            raise ValidationError("Label must be 0 (background) or 1 (foreground).")

    @validates("x")
    @validates("y")
    def validate_coordinates(self, value):
        if not (0 <= value <= 1):
            raise ValidationError("Coordinates must be between 0 and 1.")


class BoxPromptSchema(Schema):
    """ Schema for validating a bounding box annotation. """
    min_x = fields.Float(required=True)
    min_y = fields.Float(required=True)
    max_x = fields.Float(required=True)
    max_y = fields.Float(required=True)

    @validates("min_x")
    @validates("min_y")
    @validates("max_x")
    @validates("max_y")
    def validate_coordinates(self, value):
        if not (0 <= value <= 1):
            raise ValidationError("Box coordinates must be between 0 and 1.")


class SegmentationRequestSchema(Schema):
    """ Schema for validating the segmentation request. """
    use_prompts = fields.Boolean(required=True)
    image_id = fields.Integer(required=True)
    point_prompts = fields.List(fields.Nested(PointPromptSchema), required=False)
    box_prompts = fields.List(fields.Nested(BoxPromptSchema), required=False)

    @validates("point_prompts")
    def validate_point_prompts(self, value):
        if not isinstance(value, list):
            raise ValidationError("point_prompts must be a list.")

    @validates("box_prompts")
    def validate_box_prompts(self, value):
        if not isinstance(value, list):
            raise ValidationError("box_prompts must be a list.")

    @validates("image_id")
    def validate_image_id(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValidationError("image_id must be a positive integer.")
        elif Images.query.filter_by(id=value).first() is None:
            raise ValidationError("image_id does not exist in the database.")


class SegmentationResponseSchema(Schema):
    """ Schema for validating the segmentation response. """
    masks = fields.List(fields.List(fields.Integer()), required=True)  # Nested list for masks
    quality = fields.List(fields.Float(), required=True)
