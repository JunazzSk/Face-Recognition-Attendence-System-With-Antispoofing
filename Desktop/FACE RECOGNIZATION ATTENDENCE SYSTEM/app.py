from flask import Flask
from blueprints.home import home_bp
from blueprints.features import features_bp
from blueprints.studentdata import studentdata_bp
from blueprints.admin_auth import admin_auth_bp
from blueprints.admin_dashboard import admin_dashboard_bp

app = Flask(__name__)
app.secret_key = '12341234'

# Register Blueprints
app.register_blueprint(home_bp, url_prefix='/')
app.register_blueprint(features_bp, url_prefix='/features')
app.register_blueprint(studentdata_bp, url_prefix='/studentdata')
app.register_blueprint(admin_auth_bp, url_prefix='/admin_auth')
app.register_blueprint(admin_dashboard_bp, url_prefix='/admin')


# Run the application
if __name__ == "__main__":
    app.run(debug=True)
