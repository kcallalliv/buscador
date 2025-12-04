# app/extensions.py
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Se inicializa en app/__init__.py con init_app(app)
limiter = Limiter(
    key_func=lambda: (  # por sia_id o IP
        (hasattr(__import__('flask').request, "args") and __import__('flask').request.args.get("sia_id"))
        or get_remote_address()
    ),
    default_limits=["100 per hour"],
)
