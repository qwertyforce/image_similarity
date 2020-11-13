import keras
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from sklearn.neighbors import NearestNeighbors
from os import listdir
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import pickle as pk

def read_img_file(f):
    img = Image.open(f)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def resize_img_to_array(img, img_shape):
    img_array = np.array(
        img.resize(
            img_shape, 
            Image.ANTIALIAS
        )
    )    
    return img_array

def get_features(f):
    img_width, img_height = 224, 224
    img = read_img_file(f)
    
    np_img = resize_img_to_array(img, img_shape=(img_width, img_height))
    expanded_img_array = np.expand_dims(np_img, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    X_conv = model.predict(preprocessed_img)
    return X_conv[0]

model = ResNet50(weights='imagenet', include_top=False,input_shape=(224, 224, 3),pooling='max')
arr=[]

path="../../../scenery/public/images"
file_names=listdir(path)
# for f in file_names:
#     print(f)
#     features=get_features(path+"/"+f)
#     print(features)
#     arr.append(features)
# arr=np.array(arr)

# pk.dump(arr, open("ResNet50.pkl","wb"))
arr=pk.load( open("ResNet50.pkl", "rb"))
knn = NearestNeighbors(n_neighbors=10,algorithm='brute',metric='euclidean')
# pca = PCA(n_components=100)
# pca.fit(arr)
# arr = pca.transform(arr)
knn.fit(arr)

test1=(get_features("./test/212.jpeg")).reshape(1,-1)
# test1 = pca.transform(test1)
distances,indices = knn.kneighbors(test1, return_distance=True)

for i in range(indices[0].size):
    print(distances[0][i])
    print(file_names[indices[0][i]])
