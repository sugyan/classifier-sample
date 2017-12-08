#!/usr/bin/python3
import os
import tempfile
import uuid
from flask import Flask, render_template, request, send_from_directory
from recognizer import Recognizer

UPLOAD_DIRECTORY = os.path.join(tempfile.gettempdir(), 'uploads')
if not os.path.exists(UPLOAD_DIRECTORY):
    os.mkdir(UPLOAD_DIRECTORY)

app = Flask(__name__)
app.recognizer = Recognizer()


def upload_file(file):
    file_name = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIRECTORY, file_name)
    file.save(file_path)
    return file_path


@app.route('/')
def get():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def post():
    file = request.files['file']
    file_path = upload_file(file)
    result = app.recognizer.run(file_path)
    return render_template('index.html', file_path=file_path, result=result)


@app.route('{}/<filename>'.format(UPLOAD_DIRECTORY))
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIRECTORY, filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 80))
    debug = port != 80
    app.run(host='0.0.0.0', port=port, debug=debug)
