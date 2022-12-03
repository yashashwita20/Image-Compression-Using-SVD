import streamlit as st
import pandas as pd
from PIL import Image
import os
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import numpy as np
import cv2
import io, os


logo = Image.open(r'ucsd-logo.jpg')

with st.sidebar:
    st.image(logo )
    choose = option_menu("Image Compression", ["SVD Compression","DCT Compression","SVD vs DCT"],
                         icons=['1-circle', '2-circle', '3-circle'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#ffffff"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )


def rebuild_img(u, sigma, v, percent):
    m = len(u)
    n = len(v)
    a = np.zeros((m, n))
 
    count = (int)(sum(sigma))
    curSum = 0
    k = 0
    while curSum <= count * percent:
        uk = u[:, k].reshape(m, 1)
        vk = v[k].reshape(1, n)
        a += sigma[k] * np.dot(uk, vk)
        curSum += sigma[k]
        k += 1
 
    a[a < 0] = 0
    a[a > 255] = 255

    return np.rint(a).astype("uint8")



if choose == "SVD Compression":
    uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])
    k = st.slider("Compression Ratio", min_value=0, max_value=100,value = 50, step=10)/100
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns( [0.5, 0.5])
        with col1:
            st.markdown('<p style="text-align: center;">Before</p>',unsafe_allow_html=True)
            st.image(image,width=300)
            
            st.markdown('<p style="text-align: center;">'+str(round(len(uploaded_file.getvalue())/1024))+' KB' +'</p>',unsafe_allow_html=True)
            
        with col2:
            st.markdown('<p style="text-align: center;">After SVD</p>',unsafe_allow_html=True)
            
            input_img = np.array(image)
            
            u, sigma, v = np.linalg.svd(input_img[:, :, 0])
            R = rebuild_img(u, sigma, v, k)
            u, sigma, v = np.linalg.svd(input_img[:, :, 1])
            G = rebuild_img(u, sigma, v, k)
            u, sigma, v = np.linalg.svd(input_img[:, :, 2])
            B = rebuild_img(u, sigma, v, k)
            restored_img = np.stack((R, G, B), 2)

            st.image(restored_img,width=300)
            
            cv2.imwrite(f"resize output.jpg", restored_img)
            st.markdown('<p style="text-align: center;">'+str(round(os.stat("resize output.jpg").st_size/1024))+' KB' +'</p>',unsafe_allow_html=True)


if choose == "DCT Compression":
    uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])
    k = st.slider("Compression Ratio", min_value=0, max_value=100,value = 50, step=10)/100
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns( [0.5, 0.5])
        with col1:
            st.markdown('<p style="text-align: center;">Before</p>',unsafe_allow_html=True)
            st.image(image,width=300)
            
            st.markdown('<p style="text-align: center;">'+str(round(len(uploaded_file.getvalue())/1024))+' KB' +'</p>',unsafe_allow_html=True)
            
        with col2:
            st.markdown('<p style="text-align: center;">After DCT</p>',unsafe_allow_html=True)
            
            input_img = np.array(image)
            
            u, sigma, v = np.linalg.svd(input_img[:, :, 0])
            R = rebuild_img(u, sigma, v, k)
            u, sigma, v = np.linalg.svd(input_img[:, :, 1])
            G = rebuild_img(u, sigma, v, k)
            u, sigma, v = np.linalg.svd(input_img[:, :, 2])
            B = rebuild_img(u, sigma, v, k)
            restored_img = np.stack((R, G, B), 2)

            st.image(restored_img,width=300)
            
            cv2.imwrite(f"resize output.jpg", restored_img)
            st.markdown('<p style="text-align: center;">'+str(round(os.stat("resize output.jpg").st_size/1024))+' KB' +'</p>',unsafe_allow_html=True)


if choose == "SVD vs DCT":
    uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])
    k = st.slider("Compression Ratio", min_value=0, max_value=100,value = 50, step=10)/100
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2,col3 = st.columns( [1, 1,1])
        with col1:
            st.markdown('<p style="text-align: center;">Before</p>',unsafe_allow_html=True)
            st.image(image,width=200)
            
            st.markdown('<p style="text-align: center;">'+str(round(len(uploaded_file.getvalue())/1024))+' KB' +'</p>',unsafe_allow_html=True)
            
        with col2:
            st.markdown('<p style="text-align: center;">SVD</p>',unsafe_allow_html=True)
            
            input_img = np.array(image)
            
            u, sigma, v = np.linalg.svd(input_img[:, :, 0])
            R = rebuild_img(u, sigma, v, k)
            u, sigma, v = np.linalg.svd(input_img[:, :, 1])
            G = rebuild_img(u, sigma, v, k)
            u, sigma, v = np.linalg.svd(input_img[:, :, 2])
            B = rebuild_img(u, sigma, v, k)
            restored_img = np.stack((R, G, B), 2)

            st.image(restored_img,width=200)
            
            cv2.imwrite(f"resize output.jpg", restored_img)
            st.markdown('<p style="text-align: center;">'+str(round(os.stat("resize output.jpg").st_size/1024))+' KB' +'</p>',unsafe_allow_html=True)
            
        with col3:
            st.markdown('<p style="text-align: center;">DCT</p>',unsafe_allow_html=True)
            
            input_img = np.array(image)
            
            u, sigma, v = np.linalg.svd(input_img[:, :, 0])
            R = rebuild_img(u, sigma, v, k)
            u, sigma, v = np.linalg.svd(input_img[:, :, 1])
            G = rebuild_img(u, sigma, v, k)
            u, sigma, v = np.linalg.svd(input_img[:, :, 2])
            B = rebuild_img(u, sigma, v, k)
            restored_img = np.stack((R, G, B), 2)

            st.image(restored_img,width=200)
            
            cv2.imwrite(f"resize output.jpg", restored_img)
            st.markdown('<p style="text-align: center;">'+str(round(os.stat("resize output.jpg").st_size/1024))+' KB' +'</p>',unsafe_allow_html=True)

