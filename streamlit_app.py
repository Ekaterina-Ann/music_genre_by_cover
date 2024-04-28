# If you are running it locally try:
# `streamlit run app.py --server.enableXsrfProtection false`


import io
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, roc_auc_score, \
                            make_scorer, classification_report, accuracy_score
from sklearn.compose import make_column_transformer

import joblib

st.title('Классификация изображений')
def load_image():
    uploaded_image = st.file_uploader(label="Загрузите изображение")#, type=["jpg", "jpeg", "png", "webp"])
    if uploaded_image is not None:
        # Open the image using PIL
        image_test = Image.open(uploaded_image)
        # Display the image
        st.image(image_test, use_column_width=True)
        return image_test
    else:
        return None
    

image_test = load_image()
    
resnet = models.resnet50( weights="IMAGENET1K_V2")
for param in resnet.parameters():
    param.requires_grad_(False)
    
modules = list(resnet.children())[:-1]
resnet = nn.Sequential(*modules)
resnet.eval()

# from torchvision.prototype.models import resnet50, ResNet50_Weights

# # New weights:
# model = resnet50(weights=ResNet50_Weights.ImageNet1K_V2)

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                            )
preprocess = transforms.Compose([transforms.Resize(224),
                                 transforms.ToTensor(),
                                 norm])








# ### Загружаем моедель
neigh = joblib.load("model.pkl")


### Напишем функцию для векторизации изображений из streamlit
def vectorize_img_test(image_test):

    image_vec = []
    img = image_test.convert('RGB')
    image_tensor = preprocess(img)
    output_tensor = resnet(image_tensor.unsqueeze(0)).flatten().numpy()
    image_vec.append(output_tensor)
    # df_img_test = pd.DataFrame(image_vec)

    return image_vec #df_img_test


df_img_test = vectorize_img_test(image_test);
preds = neigh.predict(df_img_test);
#preds = pd.DataFrame(neigh.classes_, neigh.predict_proba(df_img_test)[0]).sort_index(ascending=False).head(3).values



result = st.button('Определить жанр')
if result:
    x = preds
    st.write('result')
    st.write(preds)



