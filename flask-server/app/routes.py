import os
from flask import Blueprint, render_template, request
from app.singleton import Classifier
from werkzeug.utils import secure_filename
import logging

UPLOAD_FOLDER = '/app/app/static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

views = Blueprint('views', __name__,
                  template_folder='templates',
                  static_folder='static')


@views.route('/', methods=['GET', 'POST'])
def classifier():
    if request.method == "POST":
        files = request.files.getlist('imageFile')
        predictions = []

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                prediction = Classifier().predict(filepath)
                predictions.append((prediction, filename))

        return render_template('index.html', predictions=predictions)    
    return render_template('index.html')
    
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS