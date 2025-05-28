import os

from services.segmentation.sam2 import SAM2Tiny, SAM2Small, SAM2Large, SAM2BasePlus
from services.segmentation import MockupSegmentationModel


class Paths:
    """ This class contains the paths to the project directories. It can be used to access the paths from
    any file in the project.
    Attributes:
        base_dir (str): The base directory of the project.
        data_dir (str): The data directory of the project.
        meso_dir (str): The directory of the meso-scale images of corals.
        micro_dir (str): The directory of the micro-scale CT Scans of corals.
        app_dir (str): Contains all folders concerning the application.
        database_dir (str): Files defining the database scheme of the app.
        routes_dir (str): The directory containing the routes of the app. This manages endpoints and requests.
        schemas_dir (str): The directory containing the schemas of endpoint request/ response data formats.
        services_dir (str): The directory containing the functionality of the app.
        weights_dir (str): The directory containing the weights of backend models.
        """
    base_dir = os.path.dirname(os.path.realpath(__file__))

    data_dir = os.path.join(base_dir, 'data')
    database = "sqlite:///" + os.path.join(data_dir, 'database.db')
    meso_dir = os.path.join(data_dir, 'meso-scale')
    images_dir = os.path.join(meso_dir, 'images')
    embedding_dir = os.path.join(meso_dir, 'embeddings')
    masks_dir = os.path.join(meso_dir, 'masks')

    micro_dir = os.path.join(data_dir, 'micro-scale')
    slices_dir = os.path.join(micro_dir, 'slices')

    app_dir = os.path.join(base_dir, 'app')
    database_dir = os.path.join(app_dir, 'database')
    routes_dir = os.path.join(app_dir, 'routes')
    schemas_dir = os.path.join(app_dir, 'schemas')
    services_dir = os.path.join(app_dir, 'services')
    segmentation_dir = os.path.join(services_dir, 'segmentation')
    weights_dir = os.path.join(segmentation_dir, 'weights')

    SAM2p1_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"


class PromptedSegmentationModelsConfig:
    """ This class contains the configuration options for the model. """
    selected_model = 'SAM2Tiny'
    available_models = {
        'Mockup': MockupSegmentationModel,
        'SAM2Tiny': SAM2Tiny,
        'SAM2Small': SAM2Small,
        'SAM2Large': SAM2Large,
        'SAM2BasePlus': SAM2BasePlus
    }


class AutomaticSegmentationModelsConfig:
    """ This class contains the configuration options for the semantic segmentation model. """
    selected_model = 'SAM2Tiny'
    available_models = {
        'Mockup': MockupSegmentationModel,
        'SAM2Tiny': SAM2Tiny,
        'SAM2Small': SAM2Small,
        'SAM2Large': SAM2Large,
        'SAM2BasePlus': SAM2BasePlus
    }
