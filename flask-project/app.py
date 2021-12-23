from flask import Flask, request  # 서버 구현을 위한 Flask 객체 import
from flask_restx import Api, Resource  # Api 구현을 위한 Api 객체 import
from werkzeug.utils import secure_filename
from flask_sqlahcmey import SQLAlchemy

import os

curpath = os.getcwd()
INPUT_FOLDER_2D = curpath+'/path/uploads2D/input'
LABEL_FOLDER_2D = curpath+'/path/uploads2D/label'
WEIGHT_FOLDER_2D = curpath+'/path/uploads2D/weight'

INPUT_FOLDER_3D = curpath+'/path/uploads3D/input'
LABEL_FOLDER_3D = curpath+'/path/uploads3D/label'
WEIGHT_FOLDER_3D = curpath+'/path/uploads3D/weight'

ALLOWED_EXTENSIONS_2D = set(['png', 'jpg', 'jpeg'])
ALLOWED_EXTENSIONS_3D = set(['nii.gz'])
ALLOWED_EXTENSIONS_weight = set(['h5'])

app = Flask(__name__)  # Flask 객체 선언, 파라미터로 어플리케이션 패키지의 이름을 넣어줌.
api = Api(app)  # Flask 객체에 Api 객체 등록

db = SQLAlchemy(app)

app.config['INPUT_FOLDER_2D'] = INPUT_FOLDER_2D
app.config['LABEL_FOLDER_2D'] = LABEL_FOLDER_2D
app.config['WEIGHT_FOLDER_2D'] = WEIGHT_FOLDER_2D

app.config['INPUT_FOLDER_3D'] = INPUT_FOLDER_3D
app.config['LABEL_FOLDER_3D'] = LABEL_FOLDER_3D
app.config['WEIGHT_FOLDER_3D'] = WEIGHT_FOLDER_3D

class Log(db.Model):
    __table_name__ = 'log'

    id = db.Column(db.Integer, primary_key=True)
    output = db.Column(db.Photo(root="/path/output/", nullable=True))
    loss = db.Colume(db.Float, nullable=True)
    dice = db.Column(db.Float, nullable=True)

def allowed_file_2D(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS_2D

def allowed_file_weight(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS_weight

def allowed_file_3D(filename):
    type_name = ".".join(filename.rsplit('.')[1:])
    return '.' in filename and \
           type_name in ALLOWED_EXTENSIONS_3D

# checked with postman
@app.route('/api/input2D', methods=['POST'])
def input2D():
    file = request.files['file']
    if file and allowed_file_2D(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['INPUT_FOLDER_2D'], filename))
        return 'Success Uploading 2D image'

@app.route('/api/label2D', methods=['POST'])
def label2D():
    file = request.files['file']
    if file and allowed_file_2D(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['LABEL_FOLDER_2D'], filename))
        return 'Success Uploading 2D label'

@app.route('/api/weight2D', methods=['POST'])
def weight2D():
    file = request.files['file']
    if file and allowed_file_weight(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['WEIGHT_FOLDER_2D'], filename))
        return 'Success Uploading 2D label'

# checked with postman
@app.route('/api/input3D', methods=['POST'])
def input3D():
    file = request.files['file']
    if file and allowed_file_3D(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER_3D'], filename))
        return 'Success Uploading 3D weight'

@app.route('/api/label3D', methods=['POST'])
def label3D():
    file = request.files['file']
    if file and allowed_file_3D(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['LABEL_FOLDER_3D'], filename))
        return 'Success Uploading 3D label'

@app.route('/api/weight3D', methods=['POST'])
def weight3D():
    file = request.files['file']
    if file and allowed_file_weight(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['WEIGHT_FOLDER_3D'], filename))
        return 'Success Uploading 3D weight'

@api.route('api/hist/all', methods=['GET'])
def get(self):
    return

@app.route('api/hist/<log_id>', methods=['GET'])
# HistoryDetail
# now, 장고처럼 구현해놓음. have to fix it.
def log(log_id):
    try:
        log = Log.objects.get(pk=log_id)
    except Log.DoesNotExist:
        #404
        return HttpResponseNotFound()
    log_json = serializer.data
    return log_json


@api.route('api/output', methods=['GET'])
def get(self):
    return

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80) # locahost
