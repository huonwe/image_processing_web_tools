import itertools
import cv2
import base64
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np

def img2b64(img,norm=False):
    img_ = img
    if norm:
        result = np.zeros(img.shape,dtype=np.float32)
        cv2.normalize(img,result,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        img_ = np.uint8(result*255.0)

    image = cv2.imencode('.png',img_)[1]
    b64_data = str(base64.b64encode(image))[2:-1]
    return b64_data

def array2b64(arr):
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.bar(np.arange(0,256),(arr / np.sum(arr,axis=0))[:,0])
    plt.xlim([0,256])
    figfile = BytesIO()
    plt.savefig(figfile,format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())
    b64 = str(figdata_png, "utf-8")
    return b64


def dftfig(img):
    rows,cols = img.shape
    nrows = cv2.getOptimalDFTSize(rows)
    ncols = cv2.getOptimalDFTSize(cols)
    nimg = np.zeros((nrows, ncols))
    nimg[:rows, :cols] = img
    img = nimg
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    return img2b64(magnitude_spectrum)


def convolve2d_vector(arr, kernel, stride=1, padding='same'):
    h, w, channel = arr.shape[0],arr.shape[1],arr.shape[2]
    k = kernel.shape[0]
    r = int(k/2)
    kernel_r = np.rot90(kernel,k=2,axes=(0,1))
    # padding outer area with 0
    padding_arr = np.zeros([h+k-1,w+k-1,channel])
    padding_arr[r:h+r,r:w+r] = arr 
    new_arr = np.zeros(arr.shape)

    vector = np.array(list(itertools.product(np.arange(r,h+r,stride),np.arange(r,w+r,stride))))
    vi = vector[:,0]
    vj = vector[:,1]  
    def _convolution(vi,vj):
        roi = padding_arr[vi-r:vi+r+1,vj-r:vj+r+1]
        new_arr[vi-r,vj-r] = np.sum(np.sum(roi*kernel_r,axis=0),axis=0)
    vfunc = np.vectorize(_convolution)    
    vfunc(vi,vj)
    return new_arr[::stride,::stride]