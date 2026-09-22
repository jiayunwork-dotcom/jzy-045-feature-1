import pytest

from app import create_app
from app.enzyme_store import EnzymeStore


@pytest.fixture()
def store_path(tmp_path):
    return str(tmp_path / "enzymes.json")


@pytest.fixture()
def store(store_path):
    return EnzymeStore(store_path)


@pytest.fixture()
def app(store_path):
    application = create_app(store_path=store_path)
    application.config.update(TESTING=True)
    return application


@pytest.fixture()
def client(app):
    return app.test_client()
