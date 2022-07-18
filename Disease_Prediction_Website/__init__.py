"""
The flask application package.
"""

from flask import Flask, flash
from flask_mail import Mail

app = Flask(__name__)
app.secret_key = "oselu"
app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'oselucarilus1@gmail.com'
app.config['MAIL_PASSWORD'] = '0703995478'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True



def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'loggedin' in session:
            return f(*args, **kwargs)
        else:
            redirect(url_for('login'))
    return wrap

import Disease_Prediction_Website.views
