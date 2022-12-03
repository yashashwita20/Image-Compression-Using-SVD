import streamlit as st
from PIL import Image
import os
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import numpy as np
import cv2
import io, os
from numpy import zeros, array, clip, trunc
from math import pi, cos, sqrt, log10
from cv2 import imread, imwrite, imshow, waitKey, destroyAllWindows, cvtColor, COLOR_BGR2RGB
import sys, time


logo = Image.open(r'4kimage_1.jpg')

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

cos_backup = array([])
def cos_values(N):
    ret = zeros((N,N))
    for n in range(len(ret)):
        for k in range(len(ret[n])):
            ret[k,n] = cos(((pi*k)*(2*n+1))/(2*N))
    global cos_backup
    cos_backup = ret
    
def direct_dct(vector):
    N = len(vector)
    if len(cos_backup) != N:
        cos_values(N)
    vector = cos_backup.dot(vector)
    vector[0] = vector[0] * sqrt(1/2)
    vector = vector * sqrt(2/N)
    return vector

def inverse_dct(vector):
    N = len(vector)
    if len(cos_backup) != N:
        cos_values(N)
    vector[0] = vector[0] * sqrt(1/2)
    vector = vector * sqrt(2/N)
    return cos_backup.T.dot(vector)

def direct_dct_2d(matrix):
    Nx,Ny = matrix.shape
    for line in range(Nx):
        matrix[line] = direct_dct(matrix[line])
    for column in range(Ny):
        matrix[:,column] = direct_dct(matrix[:,column])
    return matrix

def inverse_dct_2d(matrix):
    Nx,Ny = matrix.shape
    for column in range(Ny):
        matrix[:,column] = inverse_dct(matrix[:,column])
    for line in range(Nx):
        matrix[line] = inverse_dct(matrix[line])
    return matrix

def direct_dct_image(img):
    if img.shape[2] == 3:
        for i in range(3):
            img[:,:,i] = direct_dct_2d(img[:,:,i])
    else:
        img[:, :, 0] = direct_dct_2d(img[:, :,0])
    return img

def inverse_dct_image(img):
    if img.shape[2] == 3:
        for i in range(3):
            img[:,:,i] = inverse_dct_2d(img[:,:,i])
    else:
        img[:, :, 0] = inverse_dct_2d(img[:, :,0])
    return img.clip(0, 255)

def remove_coeficients_from_image(img, keep):
    img_new = np.zeros(img.shape)

    for i in range(keep * 3): # * 3, because 3 color channels
        index = np.unravel_index(np.absolute(img).argmax(), img.shape)
        img_new[index] = img[index] # copy it over to new image
        img[index] = 0 # remove from original so we don't count it again
        
    return img_new

# def compression_time():
#     return time.time() - start_time

#function to calculate compression ratio
def compression_ratio(original_img_arr, compressed_img_arr):
   
    compressed_size = sys.getsizeof(compressed_img_arr)
    original_size = sys.getsizeof(original_img_arr)
    compression_ratio = original_size/compressed_size

    return compression_ratio

#function to calculate Mean Square Error
def get_mse(original_img_arr, decoded_img_arr):
    
    mse = np.sum((original_img_arr.astype("float") - decoded_img_arr.astype("float")) ** 2)
    mse /= float(original_img_arr.shape[0] * decoded_img_arr.shape[1])
    
    return mse

#function to calculate Peak Signal to Noise Ratio
def get_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal ==> PSNR has no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

#function to calculate Structural Similarity Index Measure
def get_ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# If SVD is selected as compression technique
if choose == "SVD Compression":
    uploaded_file = st.file_uploader("", type=['jpg','png','jpeg']) # accepted image extensions are 'jpg','png' and 'jpeg'
    
    k = st.slider("Compression Ratio", min_value=0, max_value=100,value = 50, step=10)/100
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns( [0.5, 0.5])
        with col1:
            st.markdown('<p style="text-align: center;">Original Image</p>',unsafe_allow_html=True)
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

        st.subheader('Performance Metrics')
        st.markdown('<ul><li>'+'Compression Ratio = '+str(round(len(uploaded_file.getvalue())/1024)/round(os.stat("resize output.jpg").st_size/1024))+'</li><li>'+'Mean Square Error = '+str(round(get_mse(input_img, restored_img),2))+'</li><li>'+'Peak Signal to Noise Ratio = '+str(round(get_psnr(input_img,restored_img),2))+'</li><li>'+'Structural Similarity Index Measure = '+str(round(get_ssim(input_img, restored_img),2))+'</li></ul>',unsafe_allow_html=True)
        #st.markdown('<ul><li>'+ 'Compression time = '+ str(compression_time()) +'</li><li>'+'Compression Ratio = '+str(round(len(uploaded_file.getvalue())/1024)/round(os.stat("resize output.jpg").st_size/1024))+'</li><li>'+'Mean Square Error = '+str(round(get_mse(input_img, restored_img),2))+'</li><li>'+'Peak Signal to Noise Ratio = '+str(round(get_psnr(input_img,restored_img),2))+'</li><li>'+'Structural Similarity Index Measure = '+str(round(get_ssim(input_img, restored_img),2))+'</li></ul>',unsafe_allow_html=True)
             
        

# If DCT is selected as compression technique
if choose == "DCT Compression":
    uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])
    
    k = st.slider("k Value", min_value=0, max_value=1000,value = 550, step=10)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns( [0.5, 0.5])
        with col1:
            st.markdown('<p style="text-align: center;">Original Image</p>',unsafe_allow_html=True)
            st.image(image,width=300)
            
            st.markdown('<p style="text-align: center;">'+str(round(len(uploaded_file.getvalue())/1024))+' KB' +'</p>',unsafe_allow_html=True)
            
        with col2:
            st.markdown('<p style="text-align: center;">After DCT</p>',unsafe_allow_html=True)
            imgname = uploaded_file.name
            
            img = imread(imgname)
           
            img = img.astype('float64')
            x = direct_dct_image(img.copy())  
            y = remove_coeficients_from_image(x.copy(), k)
            comp = cvtColor(inverse_dct_image(y).astype('uint8'), COLOR_BGR2RGB)

            st.image(comp,width=300)
            
            cv2.imwrite(f"resize output.jpg", comp)
            st.markdown('<p style="text-align: center;">'+str(round(os.stat("resize output.jpg").st_size/1024))+' KB' +'</p>',unsafe_allow_html=True)
        
        st.subheader('Performance Metrics')
        st.markdown('<ul><li>'+'Compression Ratio = '+str(round(compression_ratio(img,comp),2))+'</li><li>'+'Mean Square Error = '+str(round(get_mse(img,comp),2))+'</li><li>'+'Peak Signal to Noise Ratio = '+str(round(get_psnr(img,comp),2))+'</li><li>'+'Structural Similarity Index Measure = '+str(round(get_ssim(img,comp),2))+'</li></ul>',unsafe_allow_html=True)
        
        
#  Comparison between SVD and DCT
if choose == "SVD vs DCT":
    uploaded_file = st.file_uploader("", type=['jpg','png','jpeg']) 
    k = st.slider("Compression Ratio", min_value=0, max_value=100,value = 50, step=10)/100
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2,col3 = st.columns([1, 1,1])
        with col1:
            st.markdown('<p style="text-align: center;">Original Image</p>',unsafe_allow_html=True)
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
            
            imgname = uploaded_file.name
            
            img = imread(imgname)
            img = img.astype('float64')
            x = direct_dct_image(img.copy())  
            y = remove_coeficients_from_image(x.copy(), k)
            comp = cvtColor(inverse_dct_image(y).astype('uint8'), COLOR_BGR2RGB)

            st.image(comp,width=300)
            
            cv2.imwrite(f"resize output.jpg", comp)
            st.markdown('<p style="text-align: center;">'+str(round(os.stat("resize output.jpg").st_size/1024))+' KB' +'</p>',unsafe_allow_html=True)