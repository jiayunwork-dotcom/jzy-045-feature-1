"""Uniform JSON error responses."""

from __future__ import annotations

from flask import jsonify


def json_error(status: int, error: str, reason: str):
    response = jsonify({"error": error, "reason": reason})
    response.status_code = status
    return response
