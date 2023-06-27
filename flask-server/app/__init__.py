import logging
from flask import Flask

from app.routes import views
from app.singleton import Classifier

def create_app():
    app = Flask(__name__, template_folder='templates')
    app.register_blueprint(views)
    configure_logging()
    load_mlobjects()
    return app


def configure_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def load_mlobjects():
    mlops = Classifier()
    mlops.load(model_path='/app/app/model')