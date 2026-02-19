from flask import Flask, flash, redirect, render_template, url_for, Response, jsonify
from ultralytics import YOLO
import cv2
import math, time, sys

app = Flask(__name__)


current_stage = "" # 存放第幾階段位置
'''
current_stage = "第一階段: 舌頭辨識"
current_stage = "第二階段: 健康度辨識"
current_stage = "第三階段: 病狀辨識"
'''

# 主頁面
@app.route('/')
def index():
    return render_template('index.html') #更改html的地方

# 第幾階段提示
@app.route('/get_stage_info')
def get_stage_info():
    global current_stage
    return jsonify({'stage': current_stage})

#-------------------------------------------------
healthyState = None
tongueManifestation = None
symptomsConfidence = None

def setHealthyState(init=None):
    global healthyState
    if init is not None:
        healthyState = init
    return healthyState  # 回傳目前的值

def setTongueManifestation(init=None):
    global tongueManifestation
    if init is not None:
        tongueManifestation = init
    return tongueManifestation  # 回傳目前的值

def setSymptomsConfidence(init=None):
    global symptomsConfidence
    if init is not None:
        symptomsConfidence = init
    return symptomsConfidence  # 回傳目前的值

#-------------------------------------------------

# 舌診辨識流程
def gen_frames():
    global current_stage
    tongue_model = YOLO("yolo-Weights/p1_tongue300v2_best.pt", "v8") 
    classNames = ["tongue"]

    huh_model = YOLO("yolo-Weights/p2_huh300_best.pt", "v8")
    huh_classNames = ["healthy","unhealthy"]

    allSymptoms_model = YOLO("yolo-Weights/p3_allSymptoms300_best.pt", "v8")
    allSymptoms_classNames = ['DHBT', 'QDBT', 'YDTS', 'YinDTS']

    # ydts_model = YOLO("yolo-Weights/p3_ydts_best.pt", "v8")
    # ydts_classNames = ["YDTS"]

    #
    catchUnhealthy = False 
    ydts_symptom_detected = False
    healthyTongue_detected = False
    unhealthyTongue_detected = False 
    huh_detected = False
    tongue_detected = False
    confidence = 0

# -------------------------------------------------------- 第一階段 -----------------------------------------------------------
    current_stage = "第一階段: 舌頭辨識"
    print("開始流程")
    # start webcam
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) 
    cap.set(3, 640)
    cap.set(4, 480)

    # 初始化
    catchUnhealthy = False #2nd不健康初始化
    ydts_symptom_detected = False
    healthyTongue_detected = False
    unhealthyTongue_detected = False 
    # huh_detected = False
    tongue_detected = False
    confidence = 0

    while True:
        success, img = cap.read()

        if not success:
            print("無法開啟鏡頭")
            break
            
        results = tongue_model(img, stream=True)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                # print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                # print("Class name -->", classNames[cls])

                # object details 字體或辨識框前端內容
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                if classNames[cls] == "tongue" and confidence >= 0.85: #判斷此物件是否為舌頭
                    tongue_detected = True
                
            if tongue_detected:
                print("1st: 偵測到舌頭了!")  
                croppedTongue = img[y1:y2, x1:x2]
                croppedTongue_img = cv2.imwrite("./static/dist/assets/webcam_pic/1st_cropped_tongue.jpg", croppedTongue, img) #儲存特定範圍截圖
                tongue_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/1st_raw_detected_tongue.jpg", img) # 儲存原始影像
                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
                tongue_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/1st_yolo_detected_tongue.jpg", img) #儲存yolo影像    
                break

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # cv2.imshow('Tongue Webcam', img)
        if cv2.waitKey(1) == ord('q') or tongue_detected : # 按下q或辨識到舌頭之後會退出迴圈
            break

    cap.release()
    cv2.destroyAllWindows() 

    print("1st結束")

    # ------------------------------------------------- 第二階段:對上一階段儲存的影像進行健康/不健康辨識 -----------------------------------------------------------
    print("2nd開始") 
    try:
        time.sleep(3)
        # 有抓到1st辨識的物件
        tongue_original_img = cv2.imread("./static/dist/assets/webcam_pic/1st_raw_detected_tongue.jpg")
        tongue_detection_output = huh_model.predict(source= tongue_original_img, conf=0.25, save=False) # save=False -> 不會儲存照片
        tongue_detection_output = tongue_detection_output[0] # 抓取辨識的框
        tongueCount = len(tongue_detection_output.boxes) #抓取辨識到的數量


        if(tongueCount != 0):     
            current_stage = "第二階段: 健康度辨識 (順開)"
            print("抓到1st辨識的物件 (順開)")    
            huh_results = huh_model(tongue_original_img, stream=True)
            for huh_r in huh_results:
                huh_boxes = huh_r.boxes

                for huh_box in huh_boxes:
                    huh_box = tongue_detection_output.boxes[0]
                    huh_cords = huh_box.xyxy[0].tolist()
                    huh_cords = [round(x) for x in huh_cords] # 辨識框x,y,w,h
                    huh_class_id = tongue_detection_output.names[huh_box.cls[0].item()] # 物件名稱
                    huh_conf = round(huh_box.conf[0].item(), 2) # 信心指數

                    #
                    x1, y1, x2, y2 = huh_box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                    # confidence
                    huh_confidence = math.ceil((huh_box.conf[0]*100))/100

                    # class name
                    huh_cls = int(huh_box.cls[0])

                    # object details 字體或辨識框前端內容
                    huh_org = [x1, y1]
                    huh_font = cv2.FONT_HERSHEY_SIMPLEX
                    huh_fontScale = 1
                    huh_color = (255, 0, 0)
                    huh_thickness = 2

                    #判斷此物件為健康or不健康
                    if huh_class_id == "healthy" and huh_confidence >= 0.5: # debug: classNames[cls] 換成 class_id
                        # returnHealthyState(huh_class_id) #靜態變數傳遞
                        setHealthyState(huh_class_id) #靜態變數傳遞
                        # returnHealthyconfidence(huh_confidence) #靜態變數傳遞
                        # print("health進if判斷..") # debug
                        healthyTongue_detected = True
                        print("1st接2nd (順開): 健康")
                        # huh_detected = True
                        # 儲存第二階段影像
                        huh_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/2nd_raw_detected_tongue.jpg", tongue_original_img) # 儲存原始影像
                        cv2.putText(tongue_original_img, huh_classNames[huh_cls], huh_org, huh_font, huh_fontScale, huh_color, huh_thickness)
                        cv2.rectangle(tongue_original_img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
                        huh_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/2nd_yolo_detected_tongue.jpg", tongue_original_img) #儲存yolo影像
                        print("第二階段:健康 ---> 結束")
                    
                    elif huh_class_id == "unhealthy" and confidence >= 0.5: # debug: classNames[cls] 換成 class_id
                        # returnHealthyState(huh_class_id) #靜態變數傳遞
                        setHealthyState(huh_class_id) #靜態變數傳遞
                        # print("unhealth進if判斷..") # debug
                        unhealthyTongue_detected = True
                        print("1st接2nd (順開): 不健康")
                        # 進行病徵辨識
                        huh_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/2nd_raw_detected_tongue.jpg", tongue_original_img) # 儲存原始影像
                        cv2.putText(tongue_original_img, huh_classNames[huh_cls], huh_org, huh_font, huh_fontScale, huh_color, huh_thickness)
                        cv2.rectangle(tongue_original_img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
                        huh_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/2nd_yolo_detected_tongue.jpg", tongue_original_img) #儲存yolo影像
                        catchUnhealthy = True ##2nd不健康初始化判斷
                
                ret, buffer = cv2.imencode('.jpg', tongue_original_img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                # cv2.imshow('Healthy or Unhealthy Webcam', img)
                if cv2.waitKey(1) == ord('q') or healthyTongue_detected or unhealthyTongue_detected : # 按下q或辨識到舌頭之後會退出迴圈
                    # sys.exit()
                    break

            cap.release()
            cv2.destroyAllWindows()
             

        # 表示沒抓到第一階段辨識的物件
        elif(tongueCount == 0):
            current_stage = "第二階段: 健康度辨識 (二開)"
            print("1st接2nd (二開)")
            # print("沒抓到1st辨識的物件")
            # 二開鏡頭
            # start webcam
            time.sleep(3)
            cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
            # cap = cv2.VideoCapture(1)
            cap.set(3, 640)
            cap.set(4, 480)

            # object classes
            classNames = ["healthy","unhealthy"]

            while True:
                success, img = cap.read()
                results = huh_model(img, stream=True)

                # coordinates
                for r in results:
                    boxes = r.boxes

                    for box in boxes:
                        # bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                        # put box in cam
                        # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                        # confidence
                        confidence = math.ceil((box.conf[0]*100))/100
                        # print("Confidence --->",confidence)

                        # class name
                        cls = int(box.cls[0])
                        # print("Class name -->", classNames[cls])

                        # object details
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2

                        #判斷此物件為健康or不健康
                        if classNames[cls] == "healthy" and confidence >= 0.85: 
                            # returnHealthyState(classNames[cls]) #靜態變數傳遞
                            setHealthyState(classNames[cls]) #靜態變數傳遞
                            # returnHealthyconfidence(confidence) #靜態變數傳遞
                            healthyTongue_detected = True
                            # current_stage = "辨識結束，趕緊查看結果吧!"
                        elif classNames[cls] == "unhealthy" and confidence >= 0.85:
                            # returnHealthyState(classNames[cls]) #靜態變數傳遞
                            setHealthyState(classNames[cls]) #靜態變數傳遞
                            unhealthyTongue_detected = True
                    
                    if healthyTongue_detected:
                        print("1st接2nd (二開):偵測到健康的舌頭了!")  
                        huh_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/2nd_raw_detected_tongue.jpg", img) # 儲存原始影像
                        cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
                        huh_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/2nd_yolo_detected_tongue.jpg", img) #儲存yolo影像
                        
                        
                    elif unhealthyTongue_detected:
                        print("1st接2nd (二開):偵測到不健康的舌頭了!")  
                        huh_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/2nd_raw_detected_tongue.jpg", img) # 儲存原始影像
                        cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
                        huh_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/2nd_yolo_detected_tongue.jpg", img) #儲存yolo影像
                        catchUnhealthy = True ##2nd不健康初始化判斷

                ret, buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                # cv2.imshow('Healthy or Unhealthy Webcam', img)
                if cv2.waitKey(1) == ord('q') or healthyTongue_detected or unhealthyTongue_detected: # 按下q或辨識到舌頭之後會退出迴圈
                    break

        cap.release()
        cv2.destroyAllWindows() 
                           
    except:
        pass

    print("2nd結束")
# -------------------------------------------------------- 第三階段: all symptoms -----------------------------------------------------------
    print("3rd all symptoms 開始") 

    try: 
        # 有抓到2nd辨識的物件
        huh_original_img = cv2.imread("./static/dist/assets/webcam_pic/2nd_raw_detected_tongue.jpg")
        huh_detection_output = allSymptoms_model.predict(source= huh_original_img, conf=0.25, save=False) # save=False -> 不會儲存照片
        huh_detection_output = huh_detection_output[0]
        huhCount = len(huh_detection_output.boxes) #抓取辨識到的數量

        # print("huh數量: ",huhCount) #debug

        if(huhCount != 0 and catchUnhealthy): # and
            current_stage = "第三階段 all symptoms: 病狀辨識 (順開)"
            # 有抓到第二階段的物件 -> 繼續第三階段辨識
            # print("抓到2nd辨識的物件..")

            allSymptoms_results = allSymptoms_model(huh_original_img, stream=True) 
            for allSymptoms_r in allSymptoms_results: 
                allSymptoms_boxes = allSymptoms_r.boxes

                for allSymptoms_box in allSymptoms_boxes:
                    allSymptoms_box = huh_detection_output.boxes[0]
                    allSymptoms_cords = allSymptoms_box.xyxy[0].tolist()
                    allSymptoms_cords = [round(x) for x in allSymptoms_cords] # 辨識框x,y,w,h
                    allSymptoms_class_id = huh_detection_output.names[allSymptoms_box.cls[0].item()] # 物件名稱
                    allSymptoms_conf = round(allSymptoms_box.conf[0].item(), 2) # 信心指數

                    #
                    x1, y1, x2, y2 = allSymptoms_box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                    # confidence
                    allSymptoms_confidence = math.ceil((allSymptoms_box.conf[0]*100))/100

                    # class name
                    allSymptoms_cls = int(allSymptoms_box.cls[0])

                    # object details 字體或辨識框前端內容
                    allSymptoms_org = [x1, y1]
                    allSymptoms_font = cv2.FONT_HERSHEY_SIMPLEX
                    allSymptoms_fontScale = 1
                    allSymptoms_color = (255, 0, 0)
                    allSymptoms_thickness = 2


                    if allSymptoms_class_id == "YDTS" and allSymptoms_confidence >= 0.5: # debug: classNames[cls] 換成 class_id
                        # returnTongueManifestation(allSymptoms_class_id) #傳遞舌象名稱的靜態變數
                        # returnSymptomsConfidence(allSymptoms_confidence) #傳遞YDTS辨識信心指數

                        setTongueManifestation(allSymptoms_class_id) #傳遞舌象名稱的靜態變數
                        setSymptomsConfidence(allSymptoms_confidence) #傳遞YDTS辨識信心指數
                        allSymptoms_symptom_detected = True
                        allSymptoms_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_raw_detected_tongue.jpg", huh_original_img) # 儲存原始影像
                        cv2.putText(huh_original_img, allSymptoms_classNames[allSymptoms_cls], allSymptoms_org, allSymptoms_font, allSymptoms_fontScale, allSymptoms_color, allSymptoms_thickness)
                        cv2.rectangle(huh_original_img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
                        allSymptoms_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_yolo_detected_tongue.jpg", huh_original_img) #儲存yolo影像

                    elif allSymptoms_class_id == "QDBT" and allSymptoms_confidence >= 0.5:
                        # returnTongueManifestation(allSymptoms_class_id) #傳遞舌象名稱的靜態變數
                        # returnSymptomsConfidence(allSymptoms_confidence) #傳遞YDTS辨識信心指數

                        setTongueManifestation(allSymptoms_class_id) #傳遞舌象名稱的靜態變數
                        setSymptomsConfidence(allSymptoms_confidence) #傳遞YDTS辨識信心指數
                        allSymptoms_symptom_detected = True
                        allSymptoms_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_raw_detected_tongue.jpg", huh_original_img) # 儲存原始影像
                        cv2.putText(huh_original_img, allSymptoms_classNames[allSymptoms_cls], allSymptoms_org, allSymptoms_font, allSymptoms_fontScale, allSymptoms_color, allSymptoms_thickness)
                        cv2.rectangle(huh_original_img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
                        allSymptoms_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_yolo_detected_tongue.jpg", huh_original_img) #儲存yolo影像

                    elif allSymptoms_class_id == "DHBT" and allSymptoms_confidence >= 0.5:
                        # returnTongueManifestation(allSymptoms_class_id) #傳遞舌象名稱的靜態變數
                        # returnSymptomsConfidence(allSymptoms_confidence) #傳遞YDTS辨識信心指數

                        setTongueManifestation(allSymptoms_class_id) #傳遞舌象名稱的靜態變數
                        setSymptomsConfidence(allSymptoms_confidence) #傳遞YDTS辨識信心指數
                        allSymptoms_symptom_detected = True
                        allSymptoms_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_raw_detected_tongue.jpg", huh_original_img) # 儲存原始影像
                        cv2.putText(huh_original_img, allSymptoms_classNames[allSymptoms_cls], allSymptoms_org, allSymptoms_font, allSymptoms_fontScale, allSymptoms_color, allSymptoms_thickness)
                        cv2.rectangle(huh_original_img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
                        allSymptoms_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_yolo_detected_tongue.jpg", huh_original_img) #儲存yolo影像
                    
                    elif allSymptoms_class_id == "YinDTS" and allSymptoms_confidence >= 0.5:
                        # returnTongueManifestation(allSymptoms_class_id) #傳遞舌象名稱的靜態變數
                        # returnSymptomsConfidence(allSymptoms_confidence) #傳遞YDTS辨識信心指數

                        setTongueManifestation(allSymptoms_class_id) #傳遞舌象名稱的靜態變數
                        setSymptomsConfidence(allSymptoms_confidence) #傳遞YDTS辨識信心指數
                        allSymptoms_symptom_detected = True
                        allSymptoms_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_raw_detected_tongue.jpg", huh_original_img) # 儲存原始影像
                        cv2.putText(huh_original_img, allSymptoms_classNames[allSymptoms_cls], allSymptoms_org, allSymptoms_font, allSymptoms_fontScale, allSymptoms_color, allSymptoms_thickness)
                        cv2.rectangle(huh_original_img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
                        allSymptoms_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_yolo_detected_tongue.jpg", huh_original_img) #儲存yolo影像
                    
                    else:
                        # print("其他舌象")
                        pass

                ret, buffer = cv2.imencode('.jpg', huh_original_img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                # cv2.imshow('Symptoms Webcam', img)
                if cv2.waitKey(1) == ord('q') or allSymptoms_symptom_detected : # 按下q或辨識到舌頭之後會退出迴圈
                    # sys.exit()
                    break

            cap.release()
            cv2.destroyAllWindows() 

                                    
        elif(huhCount == 0 and catchUnhealthy): # huhCount == 0 and catchUnhealthy
        # 沒抓到第二階段的物件 -> 開鏡頭
            current_stage = "第三階段 allSymptoms : 病狀辨識 (二開)"
            print("2nd接3rd (二開)")
            time.sleep(3)
            cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
            cap.set(3, 640)
            cap.set(4, 480)
            classNames = ['DHBT', 'QDBT', 'YDTS', 'YinDTS']
            while True:
                success, img = cap.read()
                results = allSymptoms_model(img, stream=True)

                # coordinates
                for r in results:
                    boxes = r.boxes

                    for box in boxes:
                        # bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                        # put box in cam
                        # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                        # confidence
                        confidence = math.ceil((box.conf[0]*100))/100
                        # print("Confidence --->",confidence)

                        # class name
                        cls = int(box.cls[0])
                        # print("Class name -->", classNames[cls])

                        # object details
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2

                        if classNames[cls] == "YDTS" and confidence >= 0.85:
                            # returnTongueManifestation(classNames[cls]) #傳遞舌象名稱的靜態變數
                            # returnSymptomsConfidence(confidence) #傳遞YDTS辨識信心指數

                            setTongueManifestation(classNames[cls]) #傳遞舌象名稱的靜態變數
                            setSymptomsConfidence(confidence) #傳遞YDTS辨識信心指數
                            allSymptoms_symptom_detected = True

                        elif classNames[cls] == "QDBT" and confidence >= 0.85:
                            # returnTongueManifestation(classNames[cls]) #傳遞舌象名稱的靜態變數
                            # returnSymptomsConfidence(confidence) #傳遞辨識信心指數

                            setTongueManifestation(classNames[cls]) #傳遞舌象名稱的靜態變數
                            setSymptomsConfidence(confidence) #傳遞YDTS辨識信心指數
                            allSymptoms_symptom_detected = True

                        elif classNames[cls] == "DHBT" and confidence >= 0.85:
                            # returnTongueManifestation(classNames[cls]) #傳遞舌象名稱的靜態變數
                            # returnSymptomsConfidence(confidence) #傳遞辨識信心指數

                            setTongueManifestation(classNames[cls]) #傳遞舌象名稱的靜態變數
                            setSymptomsConfidence(confidence) #傳遞YDTS辨識信心指數
                            allSymptoms_symptom_detected = True

                        elif classNames[cls] == "YinDTS" and confidence >= 0.85:
                            # returnTongueManifestation(classNames[cls]) #傳遞舌象名稱的靜態變數
                            # returnSymptomsConfidence(confidence) #傳遞辨識信心指數

                            setTongueManifestation(classNames[cls]) #傳遞舌象名稱的靜態變數
                            setSymptomsConfidence(confidence) #傳遞YDTS辨識信心指數
                            allSymptoms_symptom_detected = True

                        # else:
                        #     # print("其他舌象")
                        #     pass
                        
                    if allSymptoms_symptom_detected:
                        print("3rd: 偵測到YDTS舌頭了! (二開)")  
                        allSymptoms_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_raw_detected_tongue.jpg", img) # 儲存原始影像
                        cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
                        allSymptoms_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_yolo_detected_tongue.jpg", img) #儲存yolo影像    
                        break 
                
                ret, buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                # cv2.imshow('Symptoms Webcam', img)
                if cv2.waitKey(1) == ord('q') or allSymptoms_symptom_detected : # 按下q或辨識到舌頭之後會退出迴圈
                    break

        cap.release()
        cv2.destroyAllWindows() 
        current_stage = "辨識結束，趕緊查看結果吧!"

    except:
        pass
    
    print("3rd all symptoms 結束")
    print("結束流程")

    # -------------------------------------------------------- 第三階段: ydts -----------------------------------------------------------
    # print("3rd開始") 

    # try: 
    #     # 有抓到2nd辨識的物件
    #     huh_original_img = cv2.imread("./static/dist/assets/webcam_pic/2nd_raw_detected_tongue.jpg")
    #     huh_detection_output = ydts_model.predict(source= huh_original_img, conf=0.25, save=False) # save=False -> 不會儲存照片
    #     huh_detection_output = huh_detection_output[0]
    #     huhCount = len(huh_detection_output.boxes) #抓取辨識到的數量

    #     # print("huh數量: ",huhCount) #debug

    #     if(huhCount != 0 and catchUnhealthy): # and
    #         current_stage = "第三階段: 病狀辨識 (順開)"
    #         # 有抓到第二階段的物件 -> 繼續第三階段辨識
    #         # print("抓到2nd辨識的物件..")

    #         ydts_results = ydts_model(huh_original_img, stream=True) 
    #         for ydts_r in ydts_results: 
    #             ydts_boxes = ydts_r.boxes

    #             for ydts_box in ydts_boxes:
    #                 ydts_box = huh_detection_output.boxes[0]
    #                 ydts_cords = ydts_box.xyxy[0].tolist()
    #                 ydts_cords = [round(x) for x in ydts_cords] # 辨識框x,y,w,h
    #                 ydts_class_id = huh_detection_output.names[ydts_box.cls[0].item()] # 物件名稱
    #                 ydts_conf = round(ydts_box.conf[0].item(), 2) # 信心指數

    #                 #
    #                 x1, y1, x2, y2 = ydts_box.xyxy[0]
    #                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

    #                 # confidence
    #                 ydts_confidence = math.ceil((ydts_box.conf[0]*100))/100

    #                 # class name
    #                 ydts_cls = int(ydts_box.cls[0])

    #                 # object details 字體或辨識框前端內容
    #                 ydts_org = [x1, y1]
    #                 ydts_font = cv2.FONT_HERSHEY_SIMPLEX
    #                 ydts_fontScale = 1
    #                 ydts_color = (255, 0, 0)
    #                 ydts_thickness = 2


    #                 if ydts_class_id == "YDTS" and ydts_confidence >= 0.5: # debug: classNames[cls] 換成 class_id
    #                     returnTongueManifestation(ydts_class_id) #傳遞舌象名稱的靜態變數
    #                     returnYDTSconfidence(ydts_confidence) #傳遞YDTS辨識信心指數
    #                     ydts_symptom_detected = True
    #                     ydts_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_raw_detected_tongue.jpg", huh_original_img) # 儲存原始影像
    #                     cv2.putText(huh_original_img, ydts_classNames[ydts_cls], ydts_org, ydts_font, ydts_fontScale, ydts_color, ydts_thickness)
    #                     cv2.rectangle(huh_original_img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
    #                     ydts_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_yolo_detected_tongue.jpg", huh_original_img) #儲存yolo影像

    #             ret, buffer = cv2.imencode('.jpg', huh_original_img)
    #             frame = buffer.tobytes()
    #             yield (b'--frame\r\n'
    #                     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    #             # cv2.imshow('Symptoms Webcam', img)
    #             if cv2.waitKey(1) == ord('q') or ydts_symptom_detected : # 按下q或辨識到舌頭之後會退出迴圈
    #                 # sys.exit()
    #                 break

    #         cap.release()
    #         cv2.destroyAllWindows() 

                                    
    #     elif(huhCount == 0 and catchUnhealthy): # huhCount == 0 and catchUnhealthy
    #     # 沒抓到第二階段的物件 -> 開鏡頭
    #         current_stage = "第三階段: 病狀辨識 (二開)"
    #         print("2nd接3rd (二開)")
    #         time.sleep(3)
    #         cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    #         cap.set(3, 640)
    #         cap.set(4, 480)
    #         classNames = ["YDTS"]
    #         while True:
    #             success, img = cap.read()
    #             results = ydts_model(img, stream=True)

    #             # coordinates
    #             for r in results:
    #                 boxes = r.boxes

    #                 for box in boxes:
    #                     # bounding box
    #                     x1, y1, x2, y2 = box.xyxy[0]
    #                     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

    #                     # put box in cam
    #                     # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

    #                     # confidence
    #                     confidence = math.ceil((box.conf[0]*100))/100
    #                     # print("Confidence --->",confidence)

    #                     # class name
    #                     cls = int(box.cls[0])
    #                     # print("Class name -->", classNames[cls])

    #                     # object details
    #                     org = [x1, y1]
    #                     font = cv2.FONT_HERSHEY_SIMPLEX
    #                     fontScale = 1
    #                     color = (255, 0, 0)
    #                     thickness = 2

    #                     if classNames[cls] == "YDTS" and confidence >= 0.85:
    #                         returnTongueManifestation(classNames[cls]) #傳遞舌象名稱的靜態變數
    #                         returnYDTSconfidence(confidence) #傳遞YDTS辨識信心指數
    #                         ydts_symptom_detected = True
                        
    #                 if ydts_symptom_detected:
    #                     print("3rd: 偵測到YDTS舌頭了! (二開)")  
    #                     ydts_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_raw_detected_tongue.jpg", img) # 儲存原始影像
    #                     cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
    #                     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
    #                     ydts_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_yolo_detected_tongue.jpg", img) #儲存yolo影像    
    #                     break 
                
    #             ret, buffer = cv2.imencode('.jpg', img)
    #             frame = buffer.tobytes()
    #             yield (b'--frame\r\n'
    #                     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    #             # cv2.imshow('Symptoms Webcam', img)
    #             if cv2.waitKey(1) == ord('q') or ydts_symptom_detected : # 按下q或辨識到舌頭之後會退出迴圈
    #                 break

    #     cap.release()
    #     cv2.destroyAllWindows() 
    #     current_stage = "辨識結束，趕緊查看結果吧!"

    # except:
    #     pass
    
    # print("3rd結束")
    # print("結束流程")

# -------------------------------------------------- def gen_frames() FINISH !! ------------------------------------------------------------------

# 辨識流程頁面
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 辨識結果頁面
@app.route('/result')
# bug: 第一階段開到底、第二階段接第三階段(順開、二開)有問題 -> 應該是網頁顯示圖片判斷的問題?
def result():
    #預設值
    img_url = ""
    healthyState_str = "健康" #健康狀態
    tongueManifestation_str = "健康舌象" #舌象名稱
    identificationConfidenceIndex = 87 #辨識信心指數 

    setHealthyState(None)
    setTongueManifestation(None)
    setSymptomsConfidence(None)

    # 使用 setXXX 函式設置值並取得目前的值
    returnHealthyState = setHealthyState()
    returnTongueManifestation = setTongueManifestation()
    returnSymptomsConfidence = setSymptomsConfidence()       

    if returnHealthyState == "healthy":
        healthyState_str = "恭喜!! 您的舌頭很健康!"
        img_url = "./static/dist/assets/webcam_pic/2nd_yolo_detected_tongue.jpg"

    elif returnHealthyState == "unhealthy":
        healthyState_str = "糟糕!! 您的舌頭看起來不太健康!"
        img_url = "./static/dist/assets/webcam_pic/3rd_allSymptoms_yolo_detected_tongue.jpg"

        if returnTongueManifestation == "YDTS":
            tongueManifestation_str = "陽虛舌象"
            identificationConfidenceIndex = returnSymptomsConfidence*100

        elif returnTongueManifestation == "QDBT":
            tongueManifestation_str = "QDBT"
            identificationConfidenceIndex = returnSymptomsConfidence*100
        
        elif returnTongueManifestation == "DHBT":
            tongueManifestation_str = "DHBT"
            identificationConfidenceIndex = returnSymptomsConfidence*100
        
        elif returnTongueManifestation == "YinDTS":
            tongueManifestation_str = "YinDTS"
            identificationConfidenceIndex = returnSymptomsConfidence*100
        
        # elif returnTongueManifestation() == "other":
        #     # print("假設以後有其他舌象...")
        #     pass
    else:
        pass
    return render_template('result.html', img_url=img_url, healthyState_str=healthyState_str, tongueManifestation_str=tongueManifestation_str, identificationConfidenceIndex=identificationConfidenceIndex)

# 辨識結果頁面(debug看照片)
@app.route('/resultPic')
def resultPic():
    return render_template('resultPic.html')

# 返回主頁面
@app.route('/tongue_identify')
def tongue_identify():
    return render_template('tongue_identify.html')

# 中醫師端頁面
@app.route('/doctorSide')
def doctorSide():
    return render_template('doctorSide.html')

# 病患端頁面
@app.route('/patientSide')
def patientSide():
    return render_template('patientSide.html')

# login 頁面
@app.route('/login')
def login():
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=False)
