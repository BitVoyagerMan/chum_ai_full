import pandas as pd
import numpy as np
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import ast
import matplotlib.pyplot as plt
from pillow_heif import register_heif_opener



class OnlineKMeans:
    def __init__(self, threshold):
        self.threshold = threshold
        self.clusters = []
    
    def predict(self, x):
        if len(self.clusters) == 0:
            return -1
        
        distances = cdist([x], self.clusters, metric='euclidean')
        min_distance = np.min(distances)
        
        if min_distance > self.threshold:
            return -1
        
        return np.argmin(distances)
    
    def update(self, x):
        cluster = self.predict(x)
        
        if cluster == -1:
            self.clusters.append(x)
            return len(self.clusters) - 1
        else:
            self.clusters[cluster] = self.clusters[cluster] * 0.7 + x * 0.3
            return cluster
online_kmeans = OnlineKMeans(threshold=0.5)
def classify_image(url):
    feature_vector = create_feature_vector(url)
    label = online_kmeans.update(feature_vector)
    return label

register_heif_opener()

def load_and_preprocess_image(url):
    response = requests.get(url)
    
    
    img = Image.open(BytesIO(response.content))

    img.thumbnail((512, 512))
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img  
base_model = tf.keras.applications.VGG19(weights='imagenet', include_top=False)

def create_feature_vector(url):
    img = load_and_preprocess_image(url)
    feature_vector = base_model.predict(img)
    feature_vector = np.reshape(feature_vector, (7*7*512))
    return feature_vector




df = pd.read_csv('Sport_Shoes.csv')
data = df[["category_id", "sub_category_id", "sub_sub_category_id", "brand_id", "image", "chum_product_id"]]
groupedData = data.groupby(["brand_id", "category_id", "sub_category_id", "sub_sub_category_id"]).agg({"image": lambda x: list(x), "chum_product_id" : lambda x: list(x)})


for everyCategory in groupedData.to_numpy():
    for everyImageUrlArray in everyCategory[0]:
        array = ast.literal_eval(everyImageUrlArray)
        print(array[0])
        print(classify_image(array[0]))
        # for everyImageUrl in array:
        #     print(classify_image(everyImageUrl))
    
        
    break


