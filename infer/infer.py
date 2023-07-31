import os
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image
from io import BytesIO
import requests
import logging


from flask import Flask, request, jsonify
import sys

def init():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    lib_path = os.environ.get('LIBPATH')
    global model
    if lib_path is None:
        lib_path = '../lib/'

    sys.path.append(lib_path)
    from digit_recognizer import DigitRecognizer
    model = DigitRecognizer()

    model_path = os.environ.get('MODELPATH')    
    if model_path is None:
        logging.warning("MODELPATH is not defined. Default value will be used.")                      
        model_path = '../model/'                

    model_url = os.environ.get("MODELURL")    
    if model_url is None:
        logging.warning("MODELURL is not defined. Default value will be used.")                      
        model_url = "http://localhost:5002/model"
    try:
        # redis에서 modelfile을 가지고 오는데
        response = requests.get(model_url)
        if response.status_code == 200:
            model_bytes = response.content
        # redis RESTAPI를 사용할 수 있는데 modelfile을 얻어올 수 없으면 종료
        else:
            logging.critical('Modelservice seems active, but modelfile is not provided. Inferservice is shutdowned.')
            exit(-1)
        model.load_state_dict(torch.load(BytesIO(model_bytes)))
        logging.info("Model is loaded successfully from redis server.")
    except Exception as e:
        # redis를 RESTAPI를 사용할 수 없으면 model/modelfile 파일을 읽어옴
        logging.warning("Modelservice seems inactive, so model will be loaded from file.")
        try:
            model.load_state_dict(torch.load(model_path + 'modelfile')) 
            logging.info("Model is loaded successfully from redis file.")
        except Exception as e:
            logging.critical("Failed to load model from file. Inferservice is shutdowned.")
            exit(-1)

    model.eval()

def infer(image_file):
    try:
        transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

        image = Image.open(image_file).convert('L')     
        image = transform(image).unsqueeze(0)           

        # 추론
        with torch.no_grad():                           
            output = model(image)                       

        # 결과 출력
        _, predicted = torch.max(output, 1)             
        return None, predicted.item()
    except Exception as e:                              
        return e, -1

init()
app = Flask(__name__)

@app.route('/', methods=['GET'])
def i_am_alive():
    return "infer service is alive", 200

@app.route('/recognize', methods=['POST'])          
def recog_image():                                  
    if 'image' not in request.files:                
        return "No image file uploaded", 400        
    image_file = request.files['image']             
    e, result = infer(image_file)
    if e == None:
        return jsonify({'result':result})     
    else:
        return f"Error recognizing image: {str(e)}", 500



if __name__ == '__main__':      
    app.run(host='0.0.0.0', port=5001)                   
