import os
from dotenv import load_dotenv

load_dotenv()
AUTOMATIC_SEGMENTATION_BACKEND_URL = os.environ.get("AUTOMATIC_SEGMENTATION_BACKEND_URL", "http://coral-automatic-segmentation-cpu")


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
    datasets_dir = os.path.join(data_dir, 'datasets')
    thumbnails_dir = os.path.join(data_dir, 'thumbnails')

    app_dir = os.path.join(base_dir, 'app')
    database_dir = os.path.join(app_dir, 'database')
    routes_dir = os.path.join(app_dir, 'routes')
    schemas_dir = os.path.join(app_dir, 'schemas')
    services_dir = os.path.join(app_dir, 'services')
    segmentation_dir = os.path.join(services_dir, 'prompted_segmentation')
    weights_dir = os.path.join(segmentation_dir, 'weights')
