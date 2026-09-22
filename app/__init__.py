"""Flask application factory for the enzyme kinetics service."""

from __future__ import annotations

import os

from flask import Flask, jsonify

from .enzyme_store import EnzymeStore
from .errors import json_error
from .routes import api
from .validation import ValidationError


def create_app(store_path: str | None = None) -> Flask:
    app = Flask(__name__)
    app.url_map.strict_slashes = False

    store_path = store_path or os.environ.get(
        "ENZYME_STORE_PATH", os.path.join("data", "enzymes.json")
    )
    app.extensions["enzyme_store"] = EnzymeStore(store_path)

    app.register_blueprint(api)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.errorhandler(ValidationError)
    def _validation_error(exc):  # pragma: no cover - routes catch locally
        return json_error(400, "validation_error", str(exc))

    @app.errorhandler(404)
    def _not_found(exc):
        return json_error(404, "not_found", "unknown endpoint")

    @app.errorhandler(405)
    def _method_not_allowed(exc):
        return json_error(405, "method_not_allowed", "HTTP method not supported on this endpoint")

    @app.errorhandler(Exception)
    def _unexpected(exc):
        # Keep the API JSON-only; never leak an HTML stack trace.
        current_app_logger = app.logger
        current_app_logger.exception("unhandled error")
        return json_error(500, "internal_error", "internal server error")

    return app
