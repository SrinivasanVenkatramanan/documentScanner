from flask import redirect, url_for, send_file
from flask import Flask, request, jsonify, flash, session
import os
import shutil
from pancard import pan_main
from aadhar import aadhar_main
import warnings
warnings.simplefilter('ignore')
from logs import logger

app = Flask(__name__)

ALLOWED_EXTENSIONS = (['pdf', 'jpg', 'jpeg', 'png', 'JPEG', 'JPG', 'PNG'])


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


UPLOAD_FOLDER = 'data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/ocr/load_data", methods=["GET", "POST"])
def load_data():
    if request.method == 'POST':
        # check if the post request has the file part
        logger.info('Loading data...')
        if os.path.isdir(UPLOAD_FOLDER):
            shutil.rmtree('data')
        if not os.path.isdir(UPLOAD_FOLDER):
            os.mkdir(UPLOAD_FOLDER)
        if 'file' not in request.files:
            logger.info('No file part')
            return jsonify('No file part')
        file = request.files['file']
        if file.filename == '':
            logger.info('No selected file')
            return jsonify('No selected file')
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
        else:
            logger.info('Unsupported File Format')
            return jsonify('Unsupported File Format')
        if 'file2' in request.files:
            file1 = request.files['file2']
            if file1.filename == '':
                logger.info('No selected file')
                return jsonify('No selected file')
            if file1 and allowed_file(file1.filename):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
                file1.save(filepath)
            else:
                logger.info('Unsupported File Format')
                return jsonify('Unsupported File Format')
        logger.info('Data Loaded Successfully')
        return 'File Uploaded Successfully'


@app.route("/ocr/extract_data", methods=["GET", "POST"])
def extract_data():
    if request.method == 'POST':
        logger.info('Extracting data...')
        dir = os.listdir(app.config['UPLOAD_FOLDER'])
        type = request.form['type']
        if type == 'Pancard':
            logger.debug('Checking Pan Card')
            image_out = pan_main(UPLOAD_FOLDER)
            if not image_out.empty:
                logger.info('Pan Card information extracted successfully')
                return jsonify(image_out.set_index('Field').to_dict()['Information'])
            else:
                return jsonify({'Error':'Please Provide PanCard as Input'})
        elif type == 'Aadharcard':
            logger.debug('Checking Aadhar Card')
            image_out = aadhar_main(UPLOAD_FOLDER)
            if not image_out.empty:
                logger.info('Aadhar Card information extracted successfully')
                return jsonify(image_out.set_index('Field').to_dict()['Information'])
            else:
                logger.info('Please Provide AadharCard as Input')
                return jsonify({'Error':'Please Provide AadharCard as Input'})
        else:
            logger.info('Please Provide Aadharcard or Pancard Spelling Correctly')
            return jsonify({'Error':'Please Provide Aadharcard or Pancard Spelling Correctly'})


if __name__ == '__main__':
    app.run()
