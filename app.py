from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import build_montages
from imutils import paths
import imutils
import numpy as np
import argparse
import random
import cv2
import os
from PIL import Image
from keras.models import load_model
from flask import Flask, request, json, redirect, url_for
from werkzeug.utils import secure_filename
from flask import render_template, jsonify
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from annoy import AnnoyIndex
import pickle
import os
import random

app = Flask(__name__, static_url_path='/static')

model_path = 'model/ranknet_mix1'

with open("model/Ranknet.def", "r") as f:
  model_json = f.read()
  model = model_from_json(model_json)

for i in os.listdir(model_path):
    if i.endswith('.h5'):
        weight_path = model_path + '/' + i

model.load_weights(weight_path)
print("Model Successfully loaded")

for i in os.listdir(model_path):
    if i.endswith('.ann'):
        annoy_path = model_path + '/' + i

annoy_model = AnnoyIndex(4096, 'angular')
annoy_model.load(annoy_path)
print('Annoy Successfully loaded')

print('load index')
with open('static/index.pkl', 'rb') as f:
    index_dic = pickle.load(f)


model._make_predict_function()


def preprocess_img(image):
    p_image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (224, 224))
    p_image = p_image.astype(K.floatx())
    p_image *= 1. / 255
    p_image = np.expand_dims(p_image, axis=0)
    return p_image

def get_batch(iterable, n=1):
    l = len(iterable)
    for ndx in tqdm(range(0, l, n)):
        yield iterable[ndx:min(ndx + n, l)]

def get_images(files):
    images = []
    for f in files:
        try:
            image = cv2.imread(f)
            image = preprocess_img(image)
            images.append(image)
        except Exception as e:
            print(e)
    return images

def get_pred(model, image):
    if model.input_shape[0]:
        op_quer = model.predict([image,image,image])
    else:
        op_quer = model.predict(image)
    return op_quer

def predict(image_name):
    pp_images = get_images([image_name])
    batch_x = np.zeros((len(pp_images), 224, 224, 3), dtype=K.floatx())
    for i, x in enumerate(pp_images):
        batch_x[i] = x
    y_pred = get_pred(model, batch_x)
    return y_pred[-1]

    
    
def search_annoy(features, k):
    filtered_indexes = annoy_model.get_nns_by_vector(features, k, search_k=1000)
    return filtered_indexes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_new():
    global non_image
    if not os.path.exists("static/file_client/"):
        os.makedirs("static/file_client/")
    img = request.files['img']
    if 'application' in str(img):
        pass
    else:
        image = Image.open(img)
        non_image = secure_filename(img.filename)
        image.save('static/file_client/' + non_image) 

    result = predict('static/file_client/' + str(non_image))
    k = int(request.form['k'])

    
    top_k = search_annoy(result, k)
    print(top_k)

    image_info = []
    for i in top_k:
        image_info.append(index_dic[i])

    image_info.append(non_image)
    return jsonify({'image_info': image_info, 'k': k})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=6000, debug=True)
