from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from logging.config import dictConfig
import logging


UPLOAD_FOLDER = './uploaded_audio'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mp4'}


dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})


app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    app.logger.info('checkin on the filename')
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    app.logger.info(request.method)
    if request.method == 'POST':
        app.logger.info('attempting to upload new file')
        if 'file' not in request.files:
            flash('No file part')
            app.logger.warning('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected files')
            app.logger.warning('No selected files')
            return redirect(request.url)
        app.logger.info(f'Is file allowed: {allowed_file(file.filename)}')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            app.logger.info(f'saved file to {filename}')
            return redirect(url_for('index', name=filename))
    return '''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method=post enctype=multipart/form-data>
            <input type=file name=file>
            <input type=submit value=Upload>
        </form>
    '''


@app.route('/')
def index():
    return '<html><body>Hello!</body></html>'
