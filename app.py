import streamlit as st
import os
from PIL import Image
import requests
from io import BytesIO
import json
import time

ROOT_DIR = "/home/rion/image_search_engine/data/test2017"

def top_k_images(top_kth, input_img):
    file_name = str(top_kth) + ".jpg"
    image_byte_array = BytesIO()
    input_img.save(image_byte_array, format='JPEG')
    image_byte_array = image_byte_array.getvalue()

    files = {'file': (file_name, image_byte_array, 'image/jpeg')}
    url = "http://127.0.0.1:8000/search_image/"
    res = requests.post(url, files=files) 

    return json.loads(res.text)


top_k = st.selectbox(
   "Select count of images to be displayed",
   (3, 6, 9, 12, 15),
   index=None,
   placeholder="Select the number ...",
)

row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)
row5 = st.columns(3)


uploaded_file = st.file_uploader("Choose a image for search")

if uploaded_file is not None:

    input_image = Image.open(uploaded_file)

    st.image(input_image, width=400)

    start = time.time()
    res = top_k_images(top_kth=top_k, input_img=input_image)
    end_time = "Time taken for search : " + str(time.time() - start) + " secs"

    st.subheader(end_time)

    i = 0
    keys=list(res.keys())  
    for col in row1 + row2 + row3 + row4 + row5:
        if i < top_k:
            img_pth = os.path.join(ROOT_DIR, keys[i])
            image = Image.open(img_pth)
            tile = col.container(height=200)
            tile.image(image=image, width=200)
            i += 1

