from flask import Blueprint, render_template

from config import Config

web_bp = Blueprint("web", __name__)


@web_bp.get("/")
def index():
    return render_template("chat.html", office_phone=Config.SCHOOL_OFFICE_PHONE)
