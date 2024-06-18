import streamlit as st
import os
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from PIL import Image

# Set page title and favicon
st.set_page_config(page_title="Fashion Recommender", page_icon="👗")

# Set app title with custom font and color
# Set app title with custom font and color
st.title('👗Fashion Recommender System👠')
st.markdown('<style>h1{color: #8B008B; font-size: 40px; font-family: Comic Sans MS, cursive, sans-serif;}</style>', unsafe_allow_html=True)


# Set background color
st.markdown('<style>body{background-color: #F5F5F5;}</style>', unsafe_allow_html=True)

# Load precomputed features and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return True
    except:
        return False

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File upload -> Save
uploaded_file = st.file_uploader("Upload an image 📷", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # File uploaded
    image_path = 'uploaded_image.jpg'
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.read())

    # Display uploaded image
    display_image = Image.open(image_path)
    st.image(display_image, caption='Uploaded Image', use_column_width=True)

    # Add some spacing
    st.write('')
    st.write('')

    # Feature extraction
    features = feature_extraction(image_path, model)

    # Recommendation
    indices = recommend(features, feature_list)

    # Show recommendations
    st.header('👗 Top Recommendations 👠')

    # Display recommendations in a horizontal scrollable container
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(filenames[indices[0][i]], caption=f'Recommendation {i+1} 💖', use_column_width=True)
            st.markdown('<style>div.captionTextStyle{color: #000000; font-size: 14px; font-family: Arial, sans-serif;}</style>', unsafe_allow_html=True)
