# If you are running it locally try:
# `streamlit run app.py --server.enableXsrfProtection false`


# import io
# import streamlit as st
# import pandas as pd
# import numpy as np
# from PIL import Image

# def load_image():
#     uploaded_file = st.file_uploader(label='Загрузите изображение')
#     if uploaded_file is not None:
#         image_data = uploaded_file.get_value()
#         st.image(image_data)
#         return Image.open(io.openBytesIO(image_data))
#     else:
#         return None

# st.title('Классификация изображений')
# img = load_image()
# result = st.button('Определить жанр')

### Импорты
import streamlit as st
from PIL import Image


import pandas as pd
import numpy as np
import re
import time
import statistics as stat
import math
import random

from sklearn.neighbors import KNeighborsClassifier



import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18
from torchvision import transforms





import sklearn
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report





import os



import random

RANDOM_STATE = 12345





from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, roc_auc_score, \
                            make_scorer, classification_report, accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline
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

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                            )
preprocess = transforms.Compose([transforms.Resize(224),
                                 transforms.ToTensor(),
                                 norm])








### Пишем функции для streamlit
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
# preds = neigh.predict(df_img_test);
preds = pd.DataFrame(neigh.classes_, neigh.predict_proba(df_img_test)[0]).sort_index(ascending=False).head(3).values



result = st.button('Определить жанр')
if result:
    x = preds
    st.write('result')
    st.write(preds)



