from flask import render_template

def init_app(app):
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/explore")
    def explore():
        return render_template("explore.html")

    @app.route("/subgroups")
    def subgroups():
        return render_template("subgroups.html")

    @app.route("/whatif")
    def whatif():
        return render_template("whatif.html")

    @app.route("/export")
    def export():
        return render_template("export.html")