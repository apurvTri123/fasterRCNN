import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from werkzeug.utils import secure_filename
import numpy as np

import torch
import torchvision

from PIL import Image
from torchvision.transforms import ToTensor

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler


ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
UPLOAD_FOLDER = 'uploads'
# Define model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model3 = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # foreground(wheat) + background
# get number of input features for the classifier
in_features = model3.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model3.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model3.load_state_dict(torch.load("./torchModel.pt"))
params = [p for p in model3.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
model3.to(device)

fasterRCNN = model3  ## Upload the saved model


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def predict(file):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    imagev = Image.open(file)
    imagev = ToTensor()(imagev).unsqueeze(0) # unsqueeze to add artificial first dimension
    imagev=imagev.to(device)
    pred = fasterRCNN(imagev)
    output = pred[0]['boxes'].data.cpu().numpy()
    return output

app = Flask(__name__, template_folder='Templates')  ## To upload files to folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def template_test():
    return render_template('home.html', label='', imagesource='file://null')  ## Routing url


@app.route('/', methods=['GET', 'POST'])  ## Main post and get methods for calling and getting a response from the server
def upload_file(): 
    if request.method == 'POST':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            output = predict(file_path)
    return render_template("home.html", label=output, imagesource=file_path)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(threaded=False)
