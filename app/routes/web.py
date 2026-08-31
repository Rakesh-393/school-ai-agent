from flask import Blueprint, render_template

web_bp = Blueprint("web", __name__)


@web_bp.get("/")
def index():
    return render_template("chat.html")
