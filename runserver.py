"""
This script runs the Disease_Prediction_Website application using a development server.
"""

from os import environ
from Disease_Prediction_Website import app


if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')
    PORT = 5555
    app.run(HOST, PORT)
