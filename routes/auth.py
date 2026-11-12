"""Authentication routes for ResearchMate."""

import logging
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
)
from flask_login import login_user, logout_user, login_required, current_user

from database.db import db
from database.models import User, ResearchHistory

logger = logging.getLogger(__name__)
auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """Researcher login page and submission handler."""
    if current_user.is_authenticated:
        return redirect(url_for("views.dashboard"))

    if request.method == "POST":
        # Support both form submissions and JSON requests
        if request.is_json:
            data = request.get_json() or {}
            identity = data.get("identity", "").strip()
            password = data.get("password", "")
            remember = bool(data.get("remember", False))
        else:
            identity = request.form.get("identity", "").strip()
            password = request.form.get("password", "")
            remember = bool(request.form.get("remember"))

        if not identity or not password:
            msg = "Please provide both username/email and password."
            if request.is_json:
                return jsonify({"success": False, "error": msg}), 400
            flash(msg, "danger")
            return render_template("login.html")

        # Find user by username or email (case-insensitive)
        user = User.query.filter(
            (User.username.ilike(identity)) | (User.email.ilike(identity))
        ).first()

        if not user or not user.check_password(password):
            msg = "Invalid username/email or password."
            if request.is_json:
                return jsonify({"success": False, "error": msg}), 401
            flash(msg, "danger")
            return render_template("login.html")

        login_user(user, remember=remember)
        logger.info(f"User {user.username} logged in successfully.")

        next_page = request.args.get("next")
        if not next_page or not next_page.startswith("/"):
            next_page = url_for("views.dashboard")

        if request.is_json:
            return jsonify({
                "success": True,
                "message": "Login successful",
                "redirect_url": next_page,
                "user": user.to_dict(),
            })

        flash(f"Welcome back, {user.full_name}!", "success")
        return redirect(next_page)

    return render_template("login.html")


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """Researcher registration page and submission handler."""
    if current_user.is_authenticated:
        return redirect(url_for("views.dashboard"))

    if request.method == "POST":
        if request.is_json:
            data = request.get_json() or {}
            username = data.get("username", "").strip()
            email = data.get("email", "").strip()
            full_name = data.get("full_name", "").strip()
            institution = data.get("institution", "").strip()
            password = data.get("password", "")
            confirm_password = data.get("confirm_password", "")
        else:
            username = request.form.get("username", "").strip()
            email = request.form.get("email", "").strip()
            full_name = request.form.get("full_name", "").strip()
            institution = request.form.get("institution", "").strip()
            password = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")

        # Validations
        if not username or not email or not password:
            msg = "Username, email, and password are required."
            if request.is_json:
                return jsonify({"success": False, "error": msg}), 400
            flash(msg, "danger")
            return render_template("register.html")

        if len(password) < 6:
            msg = "Password must be at least 6 characters long."
            if request.is_json:
                return jsonify({"success": False, "error": msg}), 400
            flash(msg, "danger")
            return render_template("register.html")

        if password != confirm_password:
            msg = "Passwords do not match."
            if request.is_json:
                return jsonify({"success": False, "error": msg}), 400
            flash(msg, "danger")
            return render_template("register.html")

        # Check existing username or email
        if User.query.filter_by(username=username).first():
            msg = "Username is already registered. Please choose another."
            if request.is_json:
                return jsonify({"success": False, "error": msg}), 400
            flash(msg, "warning")
            return render_template("register.html")

        if User.query.filter_by(email=email).first():
            msg = "Email address is already in use."
            if request.is_json:
                return jsonify({"success": False, "error": msg}), 400
            flash(msg, "warning")
            return render_template("register.html")

        new_user = User(
            username=username,
            email=email,
            full_name=full_name or "Academic Researcher",
            institution=institution or "AICTE - IBM SkillsBuild",
            role="researcher",
        )
        new_user.set_password(password)

        db.session.add(new_user)
        db.session.commit()

        login_user(new_user)
        logger.info(f"New user registered: {new_user.username}")

        if request.is_json:
            return jsonify({
                "success": True,
                "message": "Account created successfully",
                "redirect_url": url_for("views.dashboard"),
                "user": new_user.to_dict(),
            })

        flash(f"Welcome to ResearchMate, {new_user.full_name}!", "success")
        return redirect(url_for("views.dashboard"))

    return render_template("register.html")


@auth_bp.route("/logout")
@login_required
def logout():
    """Log out current user."""
    logout_user()
    flash("You have been successfully logged out.", "info")
    return redirect(url_for("auth.login"))
