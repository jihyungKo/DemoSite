from flask import Flask, request  # 서버 구현을 위한 Flask 객체 import
from flask_restx import Api, Resource  # Api 구현을 위한 Api 객체 import
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy


# for pytorch
import io
import torchvision.transforms as transforms
from PIL import Image

import lib.inference as infer
import lib.models as models

import os

curpath = os.getcwd()
INPUT_FOLDER_2D = curpath+'/path/uploads2D/input'
LABEL_FOLDER_2D = curpath+'/path/uploads2D/label'
TRAINED_FOLDER_2D = curpath+'/path/uploads2D/trained'

INPUT_FOLDER_3D = curpath+'/path/uploads3D/input'
LABEL_FOLDER_3D = curpath+'/path/uploads3D/label'
TRAINED_FOLDER_3D = curpath+'/path/uploads3D/trained'

ALLOWED_EXTENSIONS_2D = set(['png', 'jpg', 'jpeg'])
ALLOWED_EXTENSIONS_3D = set(['nii.gz'])
ALLOWED_EXTENSIONS_TRAINED = set(['pth'])

app = Flask(__name__)  # Flask 객체 선언, 파라미터로 어플리케이션 패키지의 이름을 넣어줌.
api = Api(app)  # Flask 객체에 Api 객체 등록

db = SQLAlchemy(app)

app.config['INPUT_FOLDER_2D'] = INPUT_FOLDER_2D
app.config['LABEL_FOLDER_2D'] = LABEL_FOLDER_2D
app.config['TRAINED_FOLDER_2D'] = TRAINED_FOLDER_2D

app.config['INPUT_FOLDER_3D'] = INPUT_FOLDER_3D
app.config['LABEL_FOLDER_3D'] = LABEL_FOLDER_3D
app.config['TRAINED_FOLDER_3D'] = TRAINED_FOLDER_3D

class Log(db.Model):
    __table_name__ = 'log'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    status = db.Column(db.String(10))
    output = db.Column(db.String(100))
    loss = db.Column(db.Float, nullable=True)
    dice = db.Column(db.Float, nullable=True)

    def __init__(self, status, output, loss, dice):
        self.output = output
        self.loss = loss
        self.dice = dice

    def __repr__(self):
        return f"<Log('{self.id}', '{self.output}', '{self.loss}'. '{self.dice}')>"

def allowed_file_2D(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS_2D

def allowed_file_trained(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS_TRAINED

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

@app.route('/api/trained2D', methods=['POST'])
def trained2D():
    file = request.files['file']
    if file and allowed_file_trained(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['TRAINED_FOLDER_2D'], filename))
        return 'Success Uploading 2D label'

# checked with postman
@app.route('/api/input3D', methods=['POST'])
def input3D():
    file = request.files['file']
    if file and allowed_file_3D(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER_3D'], filename))
        return 'Success Uploading 3D TRAINED'

@app.route('/api/label3D', methods=['POST'])
def label3D():
    file = request.files['file']
    if file and allowed_file_3D(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['LABEL_FOLDER_3D'], filename))
        return 'Success Uploading 3D label'

@app.route('/api/trained3D', methods=['POST'])
def trained3D():
    file = request.files['file']
    if file and allowed_file_trained(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['TRAINED_FOLDER_3D'], filename))
        return 'Success Uploading 3D TRAINED'

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

@app.route('/api/inference/<input_id>/<label_id>/<model_id>', methods=['POST'])
def inference(input_id, label_id, model_id):
    output = None
    status = "Pending"
    dice = 0
    loss = 0

    dimension = '3D'
    input_path = './path/uploads'+'{}'.format(dimension)+'/input/'+input_id+'.nii.gz'
    label_path = './path/uploads'+'{}'.format(dimension)+'/label/'+label_id+'.nii.gz'
    model_path = './path/uploads'+'{}'.format(dimension)+'/trained/'+model_id+'.pth'

    output_path = "./path/output/"+'{}'.format(dimension)+'/'

    infer.execute(input_path, label_path, model_path, output_path)
    db.session.add(Log(status, output, loss, dice))
    return "Inference Success"

# def inference(): ~~~

@app.route('/api/hist/all', methods=['GET'])
def hist_get(self):
    logs = Log.query.all()
    
    return "Get History All Success"

@app.errorhandler(404)
def page_not_found(error):
    return "404에러(Page_not_found)"

@app.route('/api/hist/<log_id>', methods=['GET'])
# HistoryDetail
def log(log_id):
    try:
        log_ = Log.query.filter(Log.id == log_id).all()
    except DoesNotExist:
        #404
        return abort(404)
    
    log_json = serializer.data
    return log_json

@app.route('/api/output', methods=['GET'])
def output_get(self):
    return

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80) # locahost