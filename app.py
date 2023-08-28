import os
from flask import Flask, request, render_template, jsonify
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from flask_sock import Sock

from utils import *

app = Flask(__name__,static_folder="./statics")
sock = Sock(app)

current_file_path = "./task3/statics/lyl.jpeg"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/img_upload",methods=["POST"])
def img_upload():
    files = request.files
    if len(files) != 1:
        return jsonify(code=100)
    # print(files.get("file"))
    file = files.get("file")
    filename = file.filename
    path = os.path.join("./task3/files",filename)
    file.save(path)
    global current_file_path
    current_file_path = path
    img = cv2.imread(current_file_path)
    print("img shape",img.shape)
    # if len(img.shape) == 2:
    #     img_gray = img
    # else:
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([img_gray],[0],None,[256],[0,256])
    hist_b64 = array2b64(hist)
    
    heq = cv2.equalizeHist(img_gray)
    heq_b64 = img2b64(heq)
    
    edges = cv2.Canny(img,100,200)
    edges_b64 = img2b64(edges)
    
    # gauss = cv2.GaussianBlur(img,[15,15],3)
    # gauss_b64 = img2b64(gauss)
    
    dft_b64 = dftfig(img_gray)
    thr, th3 = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return jsonify(code=200,hist_b64=hist_b64,heq_b64=heq_b64,edges_b64=edges_b64,dft_b64=dft_b64,otsu=thr)

@sock.route("/img/global_thr")
def global_thr(sock):
    global current_file_path
    this_file_path = current_file_path
    img = cv2.imread(this_file_path)
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img_eh = cv2.equalizeHist(img_gray)
    Canny1 = 100
    Canny2 = 200
    GS1 = 15
    GS2 = 3
    MASK_K = 4.5
    while True:
        data = sock.receive()
        if this_file_path != current_file_path:
            this_file_path = current_file_path
            img = cv2.imread(this_file_path)
            img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            img_eh = cv2.equalizeHist(img_gray)
        PREFIX = data[:3]
        if PREFIX == "THR":
            # print(current_file_path)
            thr = int(data[3:])
            
            _, th1 = cv2.threshold(img_gray,int(thr),1,cv2.THRESH_BINARY)
            _, th2 = cv2.threshold(img_gray,int(thr),1,cv2.THRESH_BINARY_INV)
            # dst = np.zeros(img.shape,dtype=np.float32)
            # dst = cv2.normalize(th,dst,alpha=0,beta=1)
            # print(dst)
            # result_1 = np.zeros(img.shape,dtype=np.float32)
            r1 = cv2.multiply(img,th1.reshape(th1.shape[0],th1.shape[1],1).repeat(3,axis=2))
            th_b64 = img2b64(r1)
            sock.send("THR1"+th_b64)
            
            r2 = cv2.multiply(img,th2.reshape(th2.shape[0],th2.shape[1],1).repeat(3,axis=2))
            th2_b64 = img2b64(r2)
            sock.send("THR2"+th2_b64)
        elif PREFIX == "CN1":
            Canny1 = int(data[3:])
            edges = cv2.Canny(img,Canny1,Canny2)
            edges_b64 = img2b64(edges)
            resp = "CANN"+edges_b64
            sock.send(resp)
        elif PREFIX == "CN2":
            Canny2 = int(data[3:])
            edges = cv2.Canny(img,Canny1,Canny2)
            edges_b64 = img2b64(edges)
            resp = "CANN"+edges_b64
            sock.send(resp)
        elif PREFIX[0:-1] == "GS":  # 高斯模板
            if PREFIX == "GS1":
                GS1 = int(data[3:])
            elif PREFIX == "GS2":
                GS2 = int(data[3:])
            elif PREFIX == "GSK":   # 非锐化MASK 系数K
                MASK_K = float(data[3:])
            gauss_blur = cv2.GaussianBlur(img,[GS1,GS1],GS2)
            gauss_b64 = img2b64(gauss_blur)
            resp = "GUS1"+gauss_b64 # 高斯模糊结果
            sock.send(resp)
            mask = cv2.subtract(img,gauss_blur)
            mask_b64 = img2b64(mask)
            resp = "GUS2" + mask_b64    # 相减结果
            sock.send(resp)
            mask_gray = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
            merge = cv2.addWeighted(img,1,mask,MASK_K,0)
            merge_gray = cv2.cvtColor(merge,cv2.COLOR_RGB2GRAY)
            merge_eh = cv2.equalizeHist(merge_gray)
            merged_b64 = img2b64(merge,norm=True)    # merge(rgb) or merge_eh(gray)
            resp = "GUS3" + merged_b64   # 合并后结果
            sock.send(resp)
        elif PREFIX == "TPL":
            if len(img.shape) == 2:
                img = img.reshape(img.shape[0],img.shape[1],1)
            h,w,c = img.shape
            
            # print("CHANNEL",c)
            params = data[3:].split(",")
            kernel = np.zeros((9,1),dtype=np.float32)
            for i,param in enumerate(params):
                try:
                    kernel[i][0] = float(param)
                except Exception:
                    continue
            kernel_sum = kernel.sum()
            # if kernel_sum < 0:
            #     continue
            if kernel_sum > 0:
                kernel = kernel / kernel_sum
            kernel = kernel.reshape(3,3)
            # print("kernel:",kernel)
            if c >= 1:
                kernel = kernel.reshape(3,3,1)
                kernel = np.repeat(kernel,c,axis=-1)
            # print("kernel shape:",kernel.shape)
            img_result = np.zeros_like(img,dtype=np.float32)
            # img_result = convolve2d_vector(img,kernel)
            # print(img_result.shape)
            # # print(img.shape)        # h, w, c
            # result = np.zeros((img_result.shape),dtype=np.float32)
            # cv2.normalize(img_result,result,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
            # r_b64 = img2b64(np.uint8(result*255.0))
            # sock.send("TPLT"+r_b64)
            img_pad = np.pad(img,((1,1),(1,1),(0,0)),'edge')
            # print(img.shape)
            # print(img_pad.shape)
            # print(img_result.shape)
            # result = np.zeros((img_result.shape),dtype=np.float32)
            for i in range(1,h+1):
                for j in range(1,w+1):
                    zone = img_pad[i-1:i+2,j-1:j+2,:]
                    # print(np.dot(zone,template).shape)
                    # print(img_result[i,j,:].shape)
                    r = np.multiply(zone,kernel)
                    # print(r.shape)
                    img_result[i-1,j-1,:] = r.sum(axis=(0,1))
                
                if i % 20 == 0:
                    # cv2.normalize(img_result,result,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
                    result_b64 = img2b64(img_result,norm=False)
                    sock.send("TPLT"+result_b64)
                    result_norm_b64 = img2b64(img_result,norm=True)
                    sock.send("TPLN"+result_norm_b64)
            # print(img_result.shape)
            # cv2.normalize(img_result,result,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
            result_b64 = img2b64(img_result,norm=False)
            sock.send("TPLT"+result_b64)
            
            result_norm_b64 = img2b64(img_result,norm=True)
            sock.send("TPLN"+result_norm_b64)
            
app.run("0.0.0.0",debug=True)