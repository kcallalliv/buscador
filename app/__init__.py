# app/__init__.py
import logging
from flask import Flask
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask import request

# -----------------------------------------------------------------------------
# Crear la app base
# -----------------------------------------------------------------------------
app = Flask(__name__)

# CORS básico (la validación estricta de orígenes se hace en common.origin_check)
CORS(app)

logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# Rate limiting (usa sia_id si viene, si no IP)
# -----------------------------------------------------------------------------
limiter = Limiter(
    key_func=lambda: (request.args.get("sia_id") or get_remote_address()),
    app=app,
    default_limits=["100 per hour"],
)

# -----------------------------------------------------------------------------
# Registrar blueprints
# -----------------------------------------------------------------------------
# Importa aquí para evitar ciclos
from app.main_prod import prod_bp  # noqa: E402
from app.main_dev import dev_bp    # noqa: E402

app.register_blueprint(prod_bp, url_prefix="/prod")
app.register_blueprint(dev_bp,  url_prefix="/dev")
