from flask import Flask
import routes

def create_app():
    app = Flask(__name__)
    routes.init_app(app)
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
