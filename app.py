from flask import Flask, render_template, request             #import
import glob
from models import EfficientNet
import PIL.Image as Image
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

app = Flask(__name__)                                    #calling

def get_model(device, classes):
    model = EfficientNet(len(classes))
    model.load_state_dict(torch.load('EfficientNet-Model.pt', map_location=torch.device(device)))
    return model

def get_transform(input_img, device):
    normalize = transforms.Normalize(
          [0.485, 0.456, 0.406], 
          [0.229, 0.224, 0.225]
    )

    test_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        normalize,
    ])
    input_img = test_transform(input_img).unsqueeze(0).to(device)
    return input_img

def load_image(img_path, device):
    img = Image.open(img_path)
    img_tensor = get_transform(img, device)
    print(img_tensor.shape)                       
    return img_tensor


def prediction(img_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = ['buildings','forest', 'glacier', 'mountain', 'sea', 'street']
    
    new_image = load_image(img_path, device)
    model = get_model(device, classes).to(device)
    model.eval()
    pred = model(new_image)
    
    if torch.cuda.is_available():
        pred = F.softmax(pred).detach().cpu().numpy()
        y_prob = pred.argmax(axis=1)[0]
    else:
        pred = F.softmax(pred).detach().numpy()
        y_prob = pred.argmax(axis=1)[0]
        
    print(pred)
    
    label = classes[y_prob]
    return label

@app.route("/", methods=['GET', 'POST'])
def home():

    return render_template('home.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    
    if request.method == 'POST':
        
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(r'Static', filename)                       #slashes should be handeled properly
        file.save(file_path)
        print(file_path)
        product = prediction(file_path)
        print(product)
        
    return render_template('predict.html', product = product, user_image = file_path)   


if __name__ == "__main__":
    app.run()                                            #run the application