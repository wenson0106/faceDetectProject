import shutil
import time
from django.http import JsonResponse
from django.shortcuts import render
import base64
import cv2
import os
import numpy as np
from PIL import Image
recognizer = cv2.face.LBPHFaceRecognizer_create()         # 啟用訓練人臉模型方法
recognizer.read('./static/face.yml')                               # 讀取人臉模型檔
cascade_path = "C:/Users/user/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"  # 載入人臉追蹤模型
face_cascade = cv2.CascadeClassifier(cascade_path)
detector = cv2.CascadeClassifier('C:/Users/user/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def index(request):
    context={}
    context['title']='face'
    return render(request,"index.html",context)

def get_video(request):
    dir=request.POST.get('name')
    video=request.FILES['video'].read()
    if not os.path.isdir('static/videos/' + dir):
        os.mkdir('static/videos/' + dir)
    else:
        shutil.rmtree('static/videos/' + dir)
        os.mkdir('static/videos/' + dir)
    FILE_OUTPUT = 'static/videos/'+dir+'/output.avi'
    #
    # # Checks and deletes the output file
    # # You cant have a existing file or it will through an error
    if os.path.isfile(FILE_OUTPUT):
        os.remove(FILE_OUTPUT)
    #
    # # opens the file 'output.avi' which is accessable as 'out_file'
    with open(FILE_OUTPUT, "wb") as out_file:  # open for [w]riting as [b]inary
        out_file.write(video)
    cap = cv2.VideoCapture(FILE_OUTPUT)
    while True:
        ret, img = cap.read()  # 讀取影片的每一幀
        if not ret:
            print("Cannot receive frame")
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 色彩轉換成黑白
        img_np = np.array(gray, 'uint8')  # 轉換成指定編碼的 numpy 陣列
        face = detector.detectMultiScale(gray)  # 擷取人臉區域
        for (x, y, w, h) in face:
            #faces.append(img_np[y:y + h, x:x + w])  # 記錄自己人臉的位置和大小內像素的數值
            im = Image.fromarray(img_np[y:y + h, x:x + w])
            i=len(os.listdir('static/videos/' + dir))
            im.save('static/videos/' + dir+"/face"+str(i)+".jpg")
            #ids.append(3)  # 記錄自己人臉對應的 id，只能是整數，都是 1 表示川普的 id
        cv2.waitKey(1)
    cap.release()
    os.remove(FILE_OUTPUT)
    return render(request, "get_video.html")


def train(request):
    context={}
    name=[]
    for dir in os.listdir("static/videos"):
        name.append(dir)

    context["name"]=name
    return render(request,"train.html",context)

def train_finished(request):
    context = {}
    recog = cv2.face.LBPHFaceRecognizer_create()

    faces = []  # 儲存人臉位置大小的串列
    ids = []  # 記錄該人臉 id 的串列
    j=0
    for dir in os.listdir("static/videos/"):
        j+=1
        i=0
        for image in os.listdir("static/videos/"+dir):
            i+=1
            img = cv2.imread("static/videos/"+dir+'/face'+str(i)+'.jpg')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_np = np.array(gray, 'uint8')
            faces.append(img_np)
            ids.append(j)
    if os.path.isfile('static/face.yml'):
        os.remove('static/face.yml')
    print('training...')  # 提示開始訓練
    recog.train(faces, np.array(ids))  # 開始訓練
    recog.save('static/face.yml')  # 訓練完成儲存為 face.yml
    global recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # 啟用訓練人臉模型方法
    recognizer.read('./static/face.yml')  # 讀取人臉模型檔
    global face_cascade
    face_cascade = cv2.CascadeClassifier(cascade_path)

    print('ok!')
    return render(request, "train_finished.html", context)


def detect(request):
    context={}

    return render(request,"detect.html",context)

def detect_ajax(request):
    context={}
    video = request.FILES['video'].read()
    if os.path.isfile("static/detect_temp/detect.avi"):
        os.remove("static/detect_temp/detect.avi")

    with open("static/detect_temp/detect.avi", "wb") as out_file:
        out_file.write(video)

    cap = cv2.VideoCapture("static/detect_temp/detect.avi")
    name={}
    ni=0
    conf=65
    id=""
    for n in os.listdir("static/videos"):
        ni += 1
        name[ni]=n
    while True:
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.resize(img, (720, 400))  # 縮小尺寸，加快辨識效率 9:5
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 轉換成黑白
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(80, 80))  # 追蹤人臉 (目的在於標記出外框)

        for (x, y, w, h) in faces:

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 標記人臉外框
            idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w])  # 取出 id 號碼以及信心指數 confidence
            if confidence <= conf:
                conf=confidence
                id=idnum

    localtime = time.strftime("%Y-%m-%d %I:%M:%S %p", time.localtime())
    if conf<60:
        return JsonResponse({
            'message': name[id],
            'time':localtime
        })
    else:
        return JsonResponse({
            'message': 'none'
        })


    # return render(request,"detect.html",context)