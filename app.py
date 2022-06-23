from flask import Flask, render_template, request             #import
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


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = prediction(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
