from flask import Flask, flash, redirect, render_template, url_for, Response, jsonify, request
from ultralytics import YOLO
import os, cv2, math, time, sys, json
import base64
from datetime import datetime

app = Flask(__name__)

# 存放第幾階段位置
current_stage = "" 
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

# getter & setter -> 取方法的變數
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


# 定義函數來保存圖像的Base64字符串到JSON文件
# def save_images_to_json(images_list, json_path):
#     with open(json_path, 'a') as json_file:
#         json.dump(images_list, json_file)

# 定義函數來保存圖像的路徑到JSON文件
# def save_images_to_json(new_image_dict, json_path):
    try:
        # 讀取已有的 JSON 數據
        with open(json_path, 'r') as json_file:
            existing_data = json.load(json_file)
    except FileNotFoundError:
        existing_data = []
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        existing_data = []
    
    # 如果JSON文件中有數據，並且格式正確，應該是列表包含一個字典
    if existing_data:
        existing_data[0]["tongue_original_img"].extend(new_image_dict["tongue_original_img"])
        existing_data[0]["tongue_yolo_img"].extend(new_image_dict["tongue_yolo_img"])
    else:
        existing_data = [new_image_dict]

    # 寫入 JSON 文件
    try:
        with open(json_path, 'w') as json_file:
            json.dump(existing_data, json_file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error writing JSON file: {e}")

def save_images_to_json(new_image_dict, json_path):
    try:
        # 讀取已有的 JSON 數據
        with open(json_path, 'r') as json_file:
            existing_data = json.load(json_file)
    except FileNotFoundError:
        existing_data = []
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        existing_data = []

    # 20241001 debug
    if existing_data:
        # 確認圖片路徑是否已經存在，避免重複存入
        for img_path in new_image_dict["tongue_original_img"]:
            if img_path not in existing_data[0]["tongue_original_img"]:
                existing_data[0]["tongue_original_img"].append(img_path)

        for img_path in new_image_dict["tongue_yolo_img"]:
            if img_path not in existing_data[0]["tongue_yolo_img"]:
                existing_data[0]["tongue_yolo_img"].append(img_path)
    else:
        existing_data = [new_image_dict]

    # 寫入 JSON 文件
    try:
        with open(json_path, 'w') as json_file:
            json.dump(existing_data, json_file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error writing JSON file: {e}")



#-------------------------------------------------
# 20241003
initial_image_count=0
updated_image_count=0
recognition_started = False  # 加入一個標誌變數來追蹤是否開始

# 舌診辨識流程
def gen_frames():

    # 20241003
    global initial_image_count, updated_image_count, recognition_started
    # recognition_started = True  # 標記已經開始
    recognition_started = False # 重新進入

    # 讀取初始影像數量
    json_path = "./static/dist/json/tongue_images_list.json"
    try:
        with open(json_path, 'r') as json_file:
            existing_data = json.load(json_file)
            initial_image_count = len(existing_data[0]["tongue_yolo_img"]) if existing_data else 0
    except FileNotFoundError:
        print("JSON file not found, initializing with 0 images.")
        initial_image_count = 0

    # print(f"初始影像數量: {initial_image_count}")

    # 20241003

    global current_stage
    tongue_model = YOLO("yolo-Weights/p1_tongue300v2_best.pt", "v8") 
    classNames = ["tongue"]

    huh_model = YOLO("yolo-Weights/p2_huh300v2_best.pt", "v8")
    huh_classNames = ["healthy","unhealthy"]

    allSymptoms_model = YOLO("yolo-Weights/p3_allSymptom300v2_best.pt", "v8")
    allSymptoms_classNames = ['DHBT', 'QDBT', 'YDTS', 'YinDTS']

    # 變數
    global catchUnhealthy, ydts_symptom_detected, healthyTongue_detected, unhealthyTongue_detected, huh_detected, tongue_detected, confidence
    catchUnhealthy = False 
    ydts_symptom_detected = False
    healthyTongue_detected = False
    unhealthyTongue_detected = False 
    huh_detected = False
    tongue_detected = False
    confidence = 0

    #時間戳記
    global timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 

    # 存放照片json
    # global tongue_images_list
    # tongue_images_list = []

    # 初始化一個字典來存儲所有圖片的路徑
    global tongue_images_dict
    tongue_images_dict = {
        "tongue_original_img": [],
        "tongue_yolo_img": []
    }


    # 存Json資料變數
    global getOriImg, getYoloImg, getHealthStateJsonData, getTongueSymptomJsonData
    getOriImg = ""
    getYoloImg = ""
    getHealthStateJsonData = ""
    getTongueSymptomJsonData = "健康舌象"
    global data

    # 全域變數
    global croppedTongue_img, tongue_original_img, tongue_yolo_img, huh_original_img, huh_yolo_img, allSymptoms_original_img, allSymptoms_yolo_img
    global croppedTongue_img_timestamp, tongue_original_img_timestamp, tongue_yolo_img_timestamp, huh_original_img_timestamp, huh_yolo_img_timestamp, allSymptoms_original_img_timestamp, allSymptoms_yolo_img_timestamp
    global huh_conf, allSymptoms_conf, ret

# -------------------------------------------------------- 第一階段 -----------------------------------------------------------
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

    current_stage = "鏡頭開啟中，請參考右方舌診辨識示意圖"
    # 休三秒
    time.sleep(3) 
    while True:
        success, img = cap.read()
        current_stage = "舌頭辨識中，請依照右邊範例圖示伸長舌頭" # current_stage = "第一階段: 舌頭辨識"


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
                # croppedTongue = img[y1:y2, x1:x2]
                # croppedTongue_img = cv2.imwrite("./static/dist/assets/webcam_pic/1st_cropped_tongue.jpg", croppedTongue, img) #儲存特定範圍截圖
                tongue_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/1st_raw_detected_tongue.jpg", img) # 儲存原始影像
                # 20241018 -> cv2.putText全部註解
                # cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
                tongue_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/1st_yolo_detected_tongue.jpg", img) #儲存yolo影像 

                # 儲存照片路徑至Json 20240929
                # original_image_path = f"./static/dist/assets/webcam_pic_timestamp/1st_raw_detected_tongue_{timestamp}.jpg"
                # yolo_image_path = f"./static/dist/assets/webcam_pic_timestamp/1st_yolo_detected_tongue_{timestamp}.jpg"

                # 將影像路徑添加到字典中
                # tongue_images_dict["tongue_original_img"].append(original_image_path)
                # tongue_images_dict["tongue_yolo_img"].append(yolo_image_path)

                # 將字典保存到JSON文件中
                # json_path = "./static/dist/json/tongue_images_list.json"
                # save_images_to_json(tongue_images_dict, json_path)

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
            current_stage = "舌頭辨識中，請依照右邊範例圖示伸長舌頭" # current_stage = "第二階段: 健康度辨識 (順開)"
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
                        huh_original_img_timestamp = cv2.imwrite(f"./static/dist/assets/webcam_pic_timestamp/2nd_raw_detected_tongue_{timestamp}.jpg", tongue_original_img) # 儲存原始影像
                        # cv2.putText(tongue_original_img, huh_classNames[huh_cls], huh_org, huh_font, huh_fontScale, huh_color, huh_thickness)
                        cv2.rectangle(tongue_original_img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
                        huh_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/2nd_yolo_detected_tongue.jpg", tongue_original_img) #儲存yolo影像
                        huh_yolo_img_timestamp = cv2.imwrite(f"./static/dist/assets/webcam_pic_timestamp/2nd_yolo_detected_tongue_{timestamp}.jpg", tongue_original_img) #儲存yolo影像
                        print("第二階段:健康 ---> 結束")

                        # 健康狀態
                        getHealthStateJsonData = "健康"
                        #
                        getOriImg = f"./static/dist/assets/webcam_pic_timestamp/2nd_raw_detected_tongue_{timestamp}.jpg"
                        getYoloImg = f"./static/dist/assets/webcam_pic_timestamp/2nd_yolo_detected_tongue_{timestamp}.jpg"

                        # 儲存照片路徑至Json
                        original_image_path = f"./static/dist/assets/webcam_pic_timestamp/2nd_raw_detected_tongue_{timestamp}.jpg"
                        yolo_image_path = f"./static/dist/assets/webcam_pic_timestamp/2nd_yolo_detected_tongue_{timestamp}.jpg"

                        # 將影像路徑添加到字典中
                        tongue_images_dict["tongue_original_img"].append(original_image_path)
                        tongue_images_dict["tongue_yolo_img"].append(yolo_image_path)

                        # 將字典保存到JSON文件中
                        json_path = "./static/dist/json/tongue_images_list.json"
                        save_images_to_json(tongue_images_dict, json_path) #
                    
                    elif huh_class_id == "unhealthy" and confidence >= 0.5: # debug: classNames[cls] 換成 class_id
                        # returnHealthyState(huh_class_id) #靜態變數傳遞
                        setHealthyState(huh_class_id) #靜態變數傳遞
                        # print("unhealth進if判斷..") # debug
                        unhealthyTongue_detected = True
                        print("1st接2nd (順開): 不健康")

                        # 進行病徵辨識
                        huh_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/2nd_raw_detected_tongue.jpg", tongue_original_img) # 儲存原始影像
                        # huh_original_img_timestamp = cv2.imwrite(f"./static/dist/assets/webcam_pic_timestamp/2nd_raw_detected_tongue_{timestamp}.jpg", tongue_original_img) # 儲存原始影像
                        # cv2.putText(tongue_original_img, huh_classNames[huh_cls], huh_org, huh_font, huh_fontScale, huh_color, huh_thickness)
                        cv2.rectangle(tongue_original_img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
                        huh_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/2nd_yolo_detected_tongue.jpg", tongue_original_img) #儲存yolo影像
                        # huh_yolo_img_timestamp = cv2.imwrite(f"./static/dist/assets/webcam_pic_timestamp/2nd_yolo_detected_tongue_{timestamp}.jpg", tongue_original_img) #儲存yolo影像
                        catchUnhealthy = True ##2nd不健康初始化判斷

                        # 健康狀態
                        getHealthStateJsonData = "不健康" 

                        # 儲存照片路徑至Json
                        # original_image_path = f"./static/dist/assets/webcam_pic_timestamp/2nd_raw_detected_tongue_{timestamp}.jpg"
                        # yolo_image_path = f"./static/dist/assets/webcam_pic_timestamp/2nd_yolo_detected_tongue_{timestamp}.jpg"
                        
                        # 將影像路徑添加到字典中
                        # tongue_images_dict["tongue_original_img"].append(original_image_path)
                        # tongue_images_dict["tongue_yolo_img"].append(yolo_image_path)

                        # 將字典保存到JSON文件中
                        # json_path = "./static/dist/json/tongue_images_list.json"
                        # save_images_to_json(tongue_images_dict, json_path)

                
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
            current_stage = "舌頭辨識中，請依照右邊範例圖示伸長舌頭" # current_stage = "第二階段: 健康度辨識 (二開)"
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

                        elif classNames[cls] == "unhealthy" and confidence >= 0.85:
                            # returnHealthyState(classNames[cls]) #靜態變數傳遞
                            setHealthyState(classNames[cls]) #靜態變數傳遞
                            unhealthyTongue_detected = True

                            
                    if healthyTongue_detected:
                        print("1st接2nd (二開):偵測到健康的舌頭了!")  
                        huh_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/2nd_raw_detected_tongue.jpg", img) # 儲存原始影像
                        huh_original_img_timestamp = cv2.imwrite(f"./static/dist/assets/webcam_pic_timestamp/2nd_raw_detected_tongue_{timestamp}.jpg", img) # 儲存原始影像
                        # cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
                        huh_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/2nd_yolo_detected_tongue.jpg", img) #儲存yolo影像
                        huh_yolo_img_timestamp = cv2.imwrite(f"./static/dist/assets/webcam_pic_timestamp/2nd_yolo_detected_tongue_{timestamp}.jpg", img) #儲存yolo影像

                        # 健康狀態
                        getHealthStateJsonData = "健康" 
                        #
                        getOriImg = f"./static/dist/assets/webcam_pic_timestamp/2nd_raw_detected_tongue_{timestamp}.jpg"
                        getYoloImg = f"./static/dist/assets/webcam_pic_timestamp/2nd_yolo_detected_tongue_{timestamp}.jpg"

                        # 儲存照片路徑至Json
                        original_image_path = f"./static/dist/assets/webcam_pic_timestamp/2nd_raw_detected_tongue_{timestamp}.jpg"
                        yolo_image_path = f"./static/dist/assets/webcam_pic_timestamp/2nd_yolo_detected_tongue_{timestamp}.jpg"
                        
                        # 將影像路徑添加到字典中
                        tongue_images_dict["tongue_original_img"].append(original_image_path)
                        tongue_images_dict["tongue_yolo_img"].append(yolo_image_path)

                        # 將字典保存到JSON文件中
                        json_path = "./static/dist/json/tongue_images_list.json"
                        save_images_to_json(tongue_images_dict, json_path)


                                                
                    elif unhealthyTongue_detected:
                        print("1st接2nd (二開):偵測到不健康的舌頭了!")  
                        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        huh_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/2nd_raw_detected_tongue.jpg", img) # 儲存原始影像
                        # huh_original_img_timestamp = cv2.imwrite(f"./static/dist/assets/webcam_pic_timestamp/2nd_raw_detected_tongue_{timestamp}.jpg", img) # 儲存原始影像
                        # cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
                        huh_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/2nd_yolo_detected_tongue.jpg", img) #儲存yolo影像
                        # huh_yolo_img_timestamp = cv2.imwrite(f"./static/dist/assets/webcam_pic_timestamp/2nd_yolo_detected_tongue_{timestamp}.jpg", img) #儲存yolo影像
                        catchUnhealthy = True ##2nd不健康初始化判斷

                        # 健康狀態
                        getHealthStateJsonData = "不健康" 

                        # 儲存照片路徑至Json
                        # original_image_path = f"./static/dist/assets/webcam_pic_timestamp/2nd_raw_detected_tongue_{timestamp}.jpg"
                        # yolo_image_path = f"./static/dist/assets/webcam_pic_timestamp/2nd_yolo_detected_tongue_{timestamp}.jpg"
                        
                        # 將影像路徑添加到字典中
                        # tongue_images_dict["tongue_original_img"].append(original_image_path)
                        # tongue_images_dict["tongue_yolo_img"].append(yolo_image_path)

                        # 將字典保存到JSON文件中
                        # json_path = "./static/dist/json/tongue_images_list.json"
                        # save_images_to_json(tongue_images_dict, json_path)


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
        time.sleep(3)
        # 有抓到2nd辨識的物件
        huh_original_img = cv2.imread("./static/dist/assets/webcam_pic/2nd_raw_detected_tongue.jpg")
        huh_detection_output = allSymptoms_model.predict(source= huh_original_img, conf=0.25, save=False) # save=False -> 不會儲存照片
        huh_detection_output = huh_detection_output[0]
        huhCount = len(huh_detection_output.boxes) #抓取辨識到的數量

        # print("huh數量: ",huhCount) #debug

        if(huhCount != 0 and catchUnhealthy): # and
            current_stage = "舌頭辨識中，請依照右邊範例圖示伸長舌頭" # current_stage = "第三階段 all symptoms: 病狀辨識 (順開)"
            # 有抓到第二階段的物件 -> 繼續第三階段辨識
            # print("抓到2nd辨識的物件..")

            allSymptoms_results = allSymptoms_model(huh_original_img, stream=True) 

            for allSymptoms_r in allSymptoms_results: 
                # print("順開:辨識病徵!!!") #
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


                    if allSymptoms_class_id == "YDTS" and allSymptoms_confidence >= 0.6: # debug: classNames[cls] 換成 class_id
                        # returnTongueManifestation(allSymptoms_class_id) #傳遞舌象名稱的靜態變數
                        # returnSymptomsConfidence(allSymptoms_confidence) #傳遞YDTS辨識信心指數

                        setTongueManifestation(allSymptoms_class_id) #傳遞舌象名稱的靜態變數
                        setSymptomsConfidence(allSymptoms_confidence) #傳遞YDTS辨識信心指數
                        allSymptoms_symptom_detected = True

                        allSymptoms_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_raw_detected_tongue.jpg", huh_original_img) # 儲存原始影像
                        allSymptoms_original_img_timestamp = cv2.imwrite(f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_raw_detected_tongue_{timestamp}.jpg", huh_original_img) # 儲存原始影像
                        # cv2.putText(huh_original_img, allSymptoms_classNames[allSymptoms_cls], allSymptoms_org, allSymptoms_font, allSymptoms_fontScale, allSymptoms_color, allSymptoms_thickness)
                        cv2.rectangle(huh_original_img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
                        allSymptoms_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_yolo_detected_tongue.jpg", huh_original_img) #儲存yolo影像
                        allSymptoms_yolo_img_timestamp = cv2.imwrite(f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_yolo_detected_tongue_{timestamp}.jpg", huh_original_img) #儲存yolo影像
                        print("存到 YDTS 的照片了!!!")
                        getTongueSymptomJsonData = "陽虛舌象"
                        #
                        getOriImg = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_raw_detected_tongue_{timestamp}.jpg"
                        getYoloImg = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_yolo_detected_tongue_{timestamp}.jpg"


                        # 儲存照片路徑至Json
                        original_image_path = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_raw_detected_tongue_{timestamp}.jpg"
                        yolo_image_path = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_yolo_detected_tongue_{timestamp}.jpg"
                        
                        # 將影像路徑添加到字典中
                        tongue_images_dict["tongue_original_img"].append(original_image_path)
                        tongue_images_dict["tongue_yolo_img"].append(yolo_image_path)

                        # 將字典保存到JSON文件中
                        json_path = "./static/dist/json/tongue_images_list.json"
                        save_images_to_json(tongue_images_dict, json_path)
                        break


                    elif allSymptoms_class_id == "QDBT" and allSymptoms_confidence >= 0.6:
                        # returnTongueManifestation(allSymptoms_class_id) #傳遞舌象名稱的靜態變數
                        # returnSymptomsConfidence(allSymptoms_confidence) #傳遞YDTS辨識信心指數

                        setTongueManifestation(allSymptoms_class_id) #傳遞舌象名稱的靜態變數
                        setSymptomsConfidence(allSymptoms_confidence) #傳遞YDTS辨識信心指數
                        allSymptoms_symptom_detected = True

                        allSymptoms_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_raw_detected_tongue.jpg", huh_original_img) # 儲存原始影像
                        allSymptoms_original_img_timestamp = cv2.imwrite(f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_raw_detected_tongue_{timestamp}.jpg", huh_original_img) # 儲存原始影像
                        # cv2.putText(huh_original_img, allSymptoms_classNames[allSymptoms_cls], allSymptoms_org, allSymptoms_font, allSymptoms_fontScale, allSymptoms_color, allSymptoms_thickness)
                        cv2.rectangle(huh_original_img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
                        allSymptoms_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_yolo_detected_tongue.jpg", huh_original_img) #儲存yolo影像
                        allSymptoms_yolo_img_timestamp = cv2.imwrite(f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_yolo_detected_tongue_{timestamp}.jpg", huh_original_img) #儲存yolo影像
                        print("存到 QDBT 的照片了!!!")
                        getTongueSymptomJsonData = "氣虛舌象"
                        #
                        getOriImg = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_raw_detected_tongue_{timestamp}.jpg"
                        getYoloImg = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_yolo_detected_tongue_{timestamp}.jpg"

                        # 儲存照片路徑至Json
                        original_image_path = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_raw_detected_tongue_{timestamp}.jpg"
                        yolo_image_path = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_yolo_detected_tongue_{timestamp}.jpg"
                        
                        # 將影像路徑添加到字典中
                        tongue_images_dict["tongue_original_img"].append(original_image_path)
                        tongue_images_dict["tongue_yolo_img"].append(yolo_image_path)

                        # 將字典保存到JSON文件中
                        json_path = "./static/dist/json/tongue_images_list.json"
                        save_images_to_json(tongue_images_dict, json_path)



                    elif allSymptoms_class_id == "DHBT" and allSymptoms_confidence >= 0.6:
                        # returnTongueManifestation(allSymptoms_class_id) #傳遞舌象名稱的靜態變數
                        # returnSymptomsConfidence(allSymptoms_confidence) #傳遞YDTS辨識信心指數

                        setTongueManifestation(allSymptoms_class_id) #傳遞舌象名稱的靜態變數
                        setSymptomsConfidence(allSymptoms_confidence) #傳遞YDTS辨識信心指數
                        allSymptoms_symptom_detected = True
                        
                        allSymptoms_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_raw_detected_tongue.jpg", huh_original_img) # 儲存原始影像
                        allSymptoms_original_img_timestamp = cv2.imwrite(f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_raw_detected_tongue_{timestamp}.jpg", huh_original_img) # 儲存原始影像
                        # cv2.putText(huh_original_img, allSymptoms_classNames[allSymptoms_cls], allSymptoms_org, allSymptoms_font, allSymptoms_fontScale, allSymptoms_color, allSymptoms_thickness)
                        cv2.rectangle(huh_original_img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
                        allSymptoms_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_yolo_detected_tongue.jpg", huh_original_img) #儲存yolo影像
                        allSymptoms_yolo_img_timestamp = cv2.imwrite(f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_yolo_detected_tongue_{timestamp}.jpg", huh_original_img) #儲存yolo影像
                        print("存到 DHBT 的照片了!!!")
                        getTongueSymptomJsonData = "濕熱舌象"
                        #
                        getOriImg = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_raw_detected_tongue_{timestamp}.jpg"
                        getYoloImg = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_yolo_detected_tongue_{timestamp}.jpg"

                        # 儲存照片路徑至Json
                        original_image_path = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_raw_detected_tongue_{timestamp}.jpg"
                        yolo_image_path = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_yolo_detected_tongue_{timestamp}.jpg"
                        
                        # 將影像路徑添加到字典中
                        tongue_images_dict["tongue_original_img"].append(original_image_path)
                        tongue_images_dict["tongue_yolo_img"].append(yolo_image_path)

                        # 將字典保存到JSON文件中
                        json_path = "./static/dist/json/tongue_images_list.json"
                        save_images_to_json(tongue_images_dict, json_path)



                    elif allSymptoms_class_id == "YinDTS" and allSymptoms_confidence >= 0.6:
                        # returnTongueManifestation(allSymptoms_class_id) #傳遞舌象名稱的靜態變數
                        # returnSymptomsConfidence(allSymptoms_confidence) #傳遞YDTS辨識信心指數

                        setTongueManifestation(allSymptoms_class_id) #傳遞舌象名稱的靜態變數
                        setSymptomsConfidence(allSymptoms_confidence) #傳遞YDTS辨識信心指數
                        allSymptoms_symptom_detected = True
                        
                        allSymptoms_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_raw_detected_tongue.jpg", huh_original_img) # 儲存原始影像
                        allSymptoms_original_img_timestamp = cv2.imwrite(f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_raw_detected_tongue_{timestamp}.jpg", huh_original_img) # 儲存原始影像
                        # cv2.putText(huh_original_img, allSymptoms_classNames[allSymptoms_cls], allSymptoms_org, allSymptoms_font, allSymptoms_fontScale, allSymptoms_color, allSymptoms_thickness)
                        cv2.rectangle(huh_original_img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
                        allSymptoms_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_yolo_detected_tongue.jpg", huh_original_img) #儲存yolo影像
                        allSymptoms_yolo_img_timestamp = cv2.imwrite(f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_yolo_detected_tongue_{timestamp}.jpg", huh_original_img) #儲存yolo影像
                        print("存到 YinDTS 的照片了!!!")
                        getTongueSymptomJsonData = "陰虛舌象"
                        #
                        getOriImg = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_raw_detected_tongue_{timestamp}.jpg"
                        getYoloImg = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_yolo_detected_tongue_{timestamp}.jpg"

                        # 儲存照片路徑至Json
                        original_image_path = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_raw_detected_tongue_{timestamp}.jpg"
                        yolo_image_path = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_yolo_detected_tongue_{timestamp}.jpg"
                        
                        # 將影像路徑添加到字典中
                        tongue_images_dict["tongue_original_img"].append(original_image_path)
                        tongue_images_dict["tongue_yolo_img"].append(yolo_image_path)

                        # 將字典保存到JSON文件中
                        json_path = "./static/dist/json/tongue_images_list.json"
                        save_images_to_json(tongue_images_dict, json_path)

                        
                    else:
                        print("順開 無病徵")

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
            current_stage = "舌頭辨識中，請依照右邊範例圖示伸長舌頭" # current_stage = "第三階段 allSymptoms : 病狀辨識 (二開)"
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

                        print("2辨識病徵!!!")
                        # classNames[cls]
                        if cls == "YDTS" and confidence >= 0.6:
                            # returnTongueManifestation(classNames[cls]) #傳遞舌象名稱的靜態變數
                            # returnSymptomsConfidence(confidence) #傳遞YDTS辨識信心指數

                            setTongueManifestation(classNames[cls]) #傳遞舌象名稱的靜態變數
                            setSymptomsConfidence(confidence) #傳遞YDTS辨識信心指數
                            allSymptoms_symptom_detected = True

                            allSymptoms_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_raw_detected_tongue.jpg", img) # 儲存原始影像
                            allSymptoms_original_img_timestamp = cv2.imwrite(f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_raw_detected_tongue_{timestamp}.jpg", img) # 儲存原始影像
                            # cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
                            allSymptoms_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_yolo_detected_tongue.jpg", img) #儲存yolo影像 
                            # allSymptoms_yolo_img_timestamp = cv2.imwrite(f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_yolo_detected_tongue_{timestamp}.jpg", img) #儲存yolo影像 

                            print("存到 YDTS 的照片了!!!")
                            getTongueSymptomJsonData = "陽虛舌象"
                            #
                            getOriImg = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_raw_detected_tongue_{timestamp}.jpg"
                            getYoloImg = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_yolo_detected_tongue_{timestamp}.jpg"

                            # 儲存照片路徑至Json
                            original_image_path = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_raw_detected_tongue_{timestamp}.jpg"
                            yolo_image_path = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_yolo_detected_tongue_{timestamp}.jpg"
                            
                            # 將影像路徑添加到字典中
                            tongue_images_dict["tongue_original_img"].append(original_image_path)
                            tongue_images_dict["tongue_yolo_img"].append(yolo_image_path)

                            # 將字典保存到JSON文件中
                            json_path = "./static/dist/json/tongue_images_list.json"
                            save_images_to_json(tongue_images_dict, json_path)



                        elif cls == "QDBT" and confidence >= 0.6:
                            # returnTongueManifestation(classNames[cls]) #傳遞舌象名稱的靜態變數
                            # returnSymptomsConfidence(confidence) #傳遞辨識信心指數

                            setTongueManifestation(classNames[cls]) #傳遞舌象名稱的靜態變數
                            setSymptomsConfidence(confidence) #傳遞YDTS辨識信心指數
                            allSymptoms_symptom_detected = True

                            allSymptoms_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_raw_detected_tongue.jpg", img) # 儲存原始影像
                            allSymptoms_original_img_timestamp = cv2.imwrite(f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_raw_detected_tongue_{timestamp}.jpg", img) # 儲存原始影像
                            # cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
                            allSymptoms_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_yolo_detected_tongue.jpg", img) #儲存yolo影像 
                            allSymptoms_yolo_img_timestamp = cv2.imwrite(f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_yolo_detected_tongue_{timestamp}.jpg", img) #儲存yolo影像 

                            print("存到 QDBT 的照片了!!!")
                            getTongueSymptomJsonData = "氣虛舌象"
                            #
                            getOriImg = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_raw_detected_tongue_{timestamp}.jpg"
                            getYoloImg = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_yolo_detected_tongue_{timestamp}.jpg"

                            # 儲存照片路徑至Json
                            original_image_path = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_raw_detected_tongue_{timestamp}.jpg"
                            yolo_image_path = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_yolo_detected_tongue_{timestamp}.jpg"
                            
                            # 將影像路徑添加到字典中
                            tongue_images_dict["tongue_original_img"].append(original_image_path)
                            tongue_images_dict["tongue_yolo_img"].append(yolo_image_path)

                            # 將字典保存到JSON文件中
                            json_path = "./static/dist/json/tongue_images_list.json"
                            save_images_to_json(tongue_images_dict, json_path)




                        elif cls == "DHBT" and confidence >= 0.6:
                            # returnTongueManifestation(classNames[cls]) #傳遞舌象名稱的靜態變數
                            # returnSymptomsConfidence(confidence) #傳遞辨識信心指數

                            setTongueManifestation(classNames[cls]) #傳遞舌象名稱的靜態變數
                            setSymptomsConfidence(confidence) #傳遞YDTS辨識信心指數
                            allSymptoms_symptom_detected = True

                            allSymptoms_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_raw_detected_tongue.jpg", img) # 儲存原始影像
                            allSymptoms_original_img_timestamp = cv2.imwrite(f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_raw_detected_tongue_{timestamp}.jpg", img) # 儲存原始影像
                            # cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
                            allSymptoms_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_yolo_detected_tongue.jpg", img) #儲存yolo影像 
                            allSymptoms_yolo_img_timestamp = cv2.imwrite(f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_yolo_detected_tongue_{timestamp}.jpg", img) #儲存yolo影像 

                            print("存到 DHBT 的照片了!!!")
                            getTongueSymptomJsonData = "濕熱舌象"
                            #
                            getOriImg = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_raw_detected_tongue_{timestamp}.jpg"
                            getYoloImg = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_yolo_detected_tongue_{timestamp}.jpg"

                            # 儲存照片路徑至Json
                            original_image_path = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_raw_detected_tongue_{timestamp}.jpg"
                            yolo_image_path = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_yolo_detected_tongue_{timestamp}.jpg"
                            
                            # 將影像路徑添加到字典中
                            tongue_images_dict["tongue_original_img"].append(original_image_path)
                            tongue_images_dict["tongue_yolo_img"].append(yolo_image_path)

                            # 將字典保存到JSON文件中
                            json_path = "./static/dist/json/tongue_images_list.json"
                            save_images_to_json(tongue_images_dict, json_path)



                        elif cls == "YinDTS" and confidence >= 0.6:
                            # returnTongueManifestation(classNames[cls]) #傳遞舌象名稱的靜態變數
                            # returnSymptomsConfidence(confidence) #傳遞辨識信心指數

                            setTongueManifestation(classNames[cls]) #傳遞舌象名稱的靜態變數
                            setSymptomsConfidence(confidence) #傳遞YDTS辨識信心指數
                            allSymptoms_symptom_detected = True

                            allSymptoms_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_raw_detected_tongue.jpg", img) # 儲存原始影像
                            allSymptoms_original_img_timestamp = cv2.imwrite(f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_raw_detected_tongue_{timestamp}.jpg", img) # 儲存原始影像
                            # cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
                            allSymptoms_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_yolo_detected_tongue.jpg", img) #儲存yolo影像 
                            allSymptoms_yolo_img_timestamp = cv2.imwrite(f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_yolo_detected_tongue_{timestamp}.jpg", img) #儲存yolo影像 

                            print("存到 YinDTS 的照片了!!!")
                            getTongueSymptomJsonData = "陰虛舌象"
                            #
                            getOriImg = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_raw_detected_tongue_{timestamp}.jpg"
                            getYoloImg = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_yolo_detected_tongue_{timestamp}.jpg"

                            # 儲存照片路徑至Json
                            original_image_path = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_raw_detected_tongue_{timestamp}.jpg"
                            yolo_image_path = f"./static/dist/assets/webcam_pic_timestamp/3rd_allSymptoms_yolo_detected_tongue_{timestamp}.jpg"
                            
                            # 將影像路徑添加到字典中
                            tongue_images_dict["tongue_original_img"].append(original_image_path)
                            tongue_images_dict["tongue_yolo_img"].append(yolo_image_path)

                            # 將字典保存到JSON文件中
                            json_path = "./static/dist/json/tongue_images_list.json"
                            save_images_to_json(tongue_images_dict, json_path)

                            
                        else:
                            print("二開 無病徵")
                        
                    # if allSymptoms_symptom_detected:
                    #     print("3rd: 偵測到YDTS舌頭了! (二開)")  
                    #     allSymptoms_original_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_raw_detected_tongue.jpg", img) # 儲存原始影像
                    #     cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                    #     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # put box in cam
                    #     allSymptoms_yolo_img = cv2.imwrite("./static/dist/assets/webcam_pic/3rd_allSymptoms_yolo_detected_tongue.jpg", img) #儲存yolo影像 
                    #     print("存到 3RD二開 的照片了!!!")   
                    #     break 
                
                ret, buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                # cv2.imshow('Symptoms Webcam', img)
                if cv2.waitKey(1) == ord('q') or allSymptoms_symptom_detected : # 按下q或辨識到舌頭之後會退出迴圈
                    break

        cap.release()
        cv2.destroyAllWindows() 
        current_stage = "辨識結束，趕緊查看詳細結果吧!"

        # ----------------------------------------------------- 辨識結束 --------------------------------------------------------

        # 新的資料
        data = {
            "OriImg": getOriImg,
            "YoloImg": getYoloImg,
            "健康狀態": getHealthStateJsonData,
            "舌象名稱": getTongueSymptomJsonData
        }

        directory = 'static/dist/json'  # 替換為你想要的路徑
        filename = 'data.json'
        file_path = os.path.join(directory, filename)

        # 確保目錄存在
        os.makedirs(directory, exist_ok=True)

        # 讀取現有的 JSON 檔案（如果存在）
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    existing_data = json.load(file)
                except json.JSONDecodeError:
                    # 如果檔案存在但不是有效的 JSON 格式，則初始化為空列表
                    existing_data = []
        else:
            # 如果檔案不存在，初始化為空列表
            existing_data = []

        # 確保資料是以列表形式追加的
        if isinstance(existing_data, list):
            existing_data.append(data)
        else:
            # 如果檔案內的格式不是列表，則將其轉換為列表並將新資料追加
            existing_data = [existing_data, data]

        # 將合併後的資料寫回 JSON 檔案
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=4)

        print(f"資料已成功追加到 {file_path}")

    except:
        pass
    
    print("3rd all symptoms 結束")
    print("結束流程")

    # 20241003
    # 結束流程時再讀取一次 JSON 檔案，檢查是否有新增影像
    # 更新影像數量
    try:
        with open(json_path, 'r') as json_file:
            existing_data = json.load(json_file)
            updated_image_count = len(existing_data[0]["tongue_yolo_img"]) if existing_data else 0
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        updated_image_count = initial_image_count

    # 判斷是否存儲影像
    if updated_image_count == initial_image_count:
        recognition_started = True  # 如果沒有存儲影像，則不設置為 True
        print("沒有成功存儲影像。")
    else:
        print("成功存儲影像。")

    return jsonify({'success': updated_image_count > initial_image_count})

# 20241003
@app.route('/get_image_counts', methods=['GET'])
def get_image_counts():
    global initial_image_count, updated_image_count, recognition_started
    print(f"Initial Image Count: {initial_image_count}")
    print(f"Updated Image Count: {updated_image_count}")
    print(f"Recognition Started: {recognition_started}")
    return jsonify({
        'initial_image_count': initial_image_count,
        'updated_image_count': updated_image_count,
        'recognition_started': recognition_started  # 返回辨識是否已經開始
    })
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

    # symptoms json -> symptoms.json
    global constitutionJSON, symptomJSON, reasonJSON, nursingMethodsJSON, linkJSON
    constitutionJSON = ""
    symptomJSON = ""
    reasonJSON = ""
    nursingMethodsJSON = ""
    linkJSON = {
        "症狀": "",
        "連結": ""
    }


    result_directory = 'static/dist/json'  # 替換為你想要的路徑
    result_filename = 'symptoms.json'
    result_file_path = os.path.join(result_directory, result_filename)

    with open(result_file_path, 'r', encoding='utf-8') as file:
        result_data = json.load(file)

    # print(result_data)

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
        # 查找體質為"健康"的項目並將結果賦值給 symptomJSON
        constitutionJSON = ""
        for item in result_data:
            if item.get("體質") == "健康":
                constitutionJSON = item["體質"]
                break

    elif returnHealthyState == "unhealthy":
        healthyState_str = "糟糕!! 您的舌頭看起來不太健康!"
        img_url = "./static/dist/assets/webcam_pic/3rd_allSymptoms_yolo_detected_tongue.jpg"

        # 陽虛舌象 YDTS
        if returnTongueManifestation == "YDTS":
            tongueManifestation_str = "陽虛舌象"
            identificationConfidenceIndex = returnSymptomsConfidence*100
            # 體質
            constitutionJSON = ""
            for item in result_data:
                if item.get("體質") == "陽虛":
                    constitutionJSON = item["體質"]
                    break
            # 症狀
            symptomJSON = ""
            for item in result_data:
                if item.get("症狀") == "質紅、點刺明顯，舌尖紅赤、白膩苔":
                    symptomJSON = item["症狀"]
                    break
            # 原因
            reasonJSON = ""
            for item in result_data:
                if item.get("原因") == "心身疲勞，腎陽虛等":
                    reasonJSON = item["原因"]
                    break
            # 調養方式
            nursingMethodsJSON = ""
            for item in result_data:
                if item.get("調養方式") == "多吃溫熱性食物，少吃寒性食物":
                    nursingMethodsJSON = '<a href="https://org.vghks.gov.tw/tmc/News_Content.aspx?n=8325FF04CC9D8BD5&sms=CF993FB444A1F180&s=D0675BEBB0C613C7" target="_blank">多吃溫熱性食物，少吃寒性食物</a>'
                    break
            # 舌診介紹連結
            linkJSON["症状"] = symptomJSON
            for item in result_data:
                if item.get("體質") == "陽虛":
                    linkJSON["連結"] = item.get("舌診介紹連結","ydts.html")
                    break

            print("Debug - linkJSON:", linkJSON)

        # 氣虛舌象 QDBT
        elif returnTongueManifestation == "QDBT":
            tongueManifestation_str = "氣虛舌象"
            identificationConfidenceIndex = returnSymptomsConfidence*100
            # 體質
            constitutionJSON = ""
            for item in result_data:
                if item.get("體質") == "氣虛":
                    constitutionJSON = item["體質"]
                    break
            # 症狀
            symptomJSON = ""
            for item in result_data:
                if item.get("症狀") == "舌淡白、齒痕明顯、舌體胖大":
                    symptomJSON = item["症狀"]
                    break
            # 原因
            reasonJSON = ""
            for item in result_data:
                if item.get("原因") == "腸胃、腎臟問題/甲狀腺機能低下症":
                    reasonJSON = item["原因"]
                    break
            # 調養方式
            nursingMethodsJSON = ""
            for item in result_data:
                if item.get("調養方式") == "多吃具有補氣作用的食物，少吃生冷、寒涼的食物":
                    nursingMethodsJSON = '<a href="https://urmart.com/blog/article/12800" target="_blank">多吃具有補氣作用的食物，少吃生冷、寒涼的食物</a>'
                    break
            # 舌診介紹連結
            linkJSON["症状"] = symptomJSON
            for item in result_data:
                if item.get("體質") == "氣虛":
                    linkJSON["連結"] = item.get("舌診介紹連結","qdbt.html")
                    break

            print("Debug - linkJSON:", linkJSON)

        # 濕熱舌象 DHBT
        elif returnTongueManifestation == "DHBT":
            tongueManifestation_str = "濕熱舌象"
            identificationConfidenceIndex = returnSymptomsConfidence*100
            # 體質
            constitutionJSON = ""
            for item in result_data:
                if item.get("體質") == "濕熱":
                    constitutionJSON = item["體質"]
                    break
            # 症狀
            symptomJSON = ""
            for item in result_data:
                if item.get("症狀") == "舌質紅、苔黃膩、有點刺":
                    symptomJSON = item["症狀"]
                    break
            # 原因
            reasonJSON = ""
            for item in result_data:
                if item.get("原因") == "腸胃狀況差、睡眠不足、抽菸等":
                    reasonJSON = item["原因"]
                    break
            # 調養方式
            nursingMethodsJSON = ""
            for item in result_data:
                if item.get("調養方式") == "多吃具有清熱利濕作用的食物，少吃油炸、油膩、高糖分食物":
                    nursingMethodsJSON = '<a href="https://blog.vitabox.com.tw/2023/02/plum-rain-season-food-moisture-remove/" target="_blank">多吃具有清熱利濕作用的食物，少吃油炸、油膩、高糖分食物</a>'
                    break
            # 舌診介紹連結
            linkJSON["症状"] = symptomJSON
            for item in result_data:
                if item.get("體質") == "濕熱":
                    linkJSON["連結"] = item.get("舌診介紹連結","dhbt.html")
                    break

            print("Debug - linkJSON:", linkJSON)

        # 陰虛舌象 YinDTS
        elif returnTongueManifestation == "YinDTS":
            tongueManifestation_str = "陰虛舌象"
            identificationConfidenceIndex = returnSymptomsConfidence*100
            # 體質
            constitutionJSON = ""
            for item in result_data:
                if item.get("體質") == "陰虛":
                    constitutionJSON = item["體質"]
                    break
            # 症狀
            symptomJSON = ""
            for item in result_data:
                if item.get("症狀") == "舌紅苔少、舌面紅赤、舌頭表面有溝紋":
                    symptomJSON = item["症狀"]
                    break
            # 原因
            reasonJSON = ""
            for item in result_data:
                if item.get("原因") == "慢性疾病，環境影響等":
                    reasonJSON = item["原因"]
                    break
            # 調養方式
            nursingMethodsJSON = ""
            for item in result_data:
                if item.get("調養方式") == "多吃具有滋陰潤燥作用的食物，避免辛辣、油膩、煎炸等刺激性食物":
                    nursingMethodsJSON = '<a href="https://www.hk01.com/%E6%95%99%E7%85%AE/806699/%E8%99%95%E6%9A%91%E9%A3%B2%E9%A3%9F2024-%E6%9A%91%E6%BF%95%E6%9C%AA%E6%95%A3%E6%98%93%E5%82%B7%E8%84%BE-%E4%B8%AD%E9%86%AB%E6%8E%A87%E9%A3%9F%E7%89%A93%E6%B9%AF%E6%B0%B4%E6%BB%8B%E9%99%B0%E6%BD%A4%E7%87%A5%E9%98%B2%E7%A7%8B%E4%B9%8F" target="_blank">多吃具有滋陰潤燥作用的食物，避免辛辣、油膩、煎炸等刺激性食物</a>'
                    break
            # 舌診介紹連結
            linkJSON["症状"] = symptomJSON
            for item in result_data:
                if item.get("體質") == "陰虛":
                    linkJSON["連結"] = item.get("舌診介紹連結","yindts.html")
                    break

            print("Debug - linkJSON:", linkJSON)
        
        # elif returnTongueManifestation() == "other":
        #     # print("假設以後有其他舌象...")
        #     pass
    else:
        pass
    return render_template('result.html', 
                           img_url=img_url, 
                           healthyState_str=healthyState_str, 
                           tongueManifestation_str=tongueManifestation_str, 
                           identificationConfidenceIndex=identificationConfidenceIndex, 
                           constitutionJSON=constitutionJSON,
                           symptomJSON=symptomJSON, 
                           reasonJSON=reasonJSON,
                           nursingMethodsJSON=nursingMethodsJSON,
                           linkJSON=linkJSON,
                           returnHealthyState=returnHealthyState)


# 
def getIdentificationData():
    # tongueManifestation_str = "健康舌象" #舌象名稱
    setTongueManifestation(None)
    returnTongueManifestation = setTongueManifestation()
    # print(returnTongueManifestation)
    return returnTongueManifestation
data = getIdentificationData()


# 辨識結果頁面(debug看照片)
@app.route('/resultPic')
def resultPic():
    return render_template('resultPic.html')

# 
@app.route('/tongue_identify')
def tongue_identify():
    return render_template('tongue_identify.html')

# 中醫師端頁面
@app.route('/doctorSide')
def doctorSide():
    json_path = "./static/dist/json/tongue_images_list.json"
    
    try:
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = []
    except json.JSONDecodeError:
        data = []
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        data = []

    if data:
        tongue_images = data[0]
    else:
        tongue_images = {"tongue_original_img": [], "tongue_yolo_img": []}

    return render_template('doctorSide.html', tongue_images=tongue_images)

# 病患端頁面
@app.route('/patientSide')
def patientSide():
    return render_template('patientSide.html')

# login 頁面
@app.route('/login')
def login():
    return render_template('login.html')

# 舌診簡介頁面
@app.route('/tongueIdentify_Intro')
def tongueIdentify_Intro():
    return render_template('tongueIdentify_Intro.html')

# 中醫師端表格
doctorSide_json_path = "./static/dist/json/doctorSide_list.json"

@app.route('/submit-form', methods=['POST'])
def submit_form():
    form_data = {
        'imgPath': request.form['imgPath'],
        'hospitalName': request.form['hospitalName'],
        'doctorName': request.form['doctorName'],
        'timestamp': request.form['timestamp'],
        'tongueType': request.form['tongueType'],
        # 'accuracy': request.form['accuracy'],
        'suggestions': request.form['suggestions']
    }

    if os.path.exists(doctorSide_json_path):
        with open(doctorSide_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    else:
        data = []

    data.append(form_data)

    with open(doctorSide_json_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    return jsonify(success=True)

# Data Collection
@app.route('/dataCollection')
def dataCollection():
    # 讀取 JSON 檔案
    with open('./static/dist/json/doctorSide_list.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    return render_template('dataCollection.html',data=data)

# Data Collection v2
@app.route('/dataCollectionNew')
def dataCollectionNew():
    # 讀取 data.json 檔案
    with open('./static/dist/json/data.json', 'r', encoding='utf-8') as file:
        ai_data = json.load(file)
    
    # 讀取 doctorSide_list.json 檔案
    with open('./static/dist/json/doctorSide_list.json', 'r', encoding='utf-8') as file:
        doctor_data = json.load(file)

    #
    # 匹配 YoloImg 與 imgPath，並提取舌象名稱和 tongueType
    matched_data = []
    for ai in ai_data:
        for doctor in doctor_data:
            if ai["YoloImg"] == doctor["imgPath"]:
                matched_data.append({
                    "YoloImg": ai["YoloImg"],
                    "舌象名稱": ai["舌象名稱"],
                    "tongueType": doctor["tongueType"],
                    "hospitalName": doctor["hospitalName"],
                    "doctorName": doctor["doctorName"],
                    "timestamp": doctor["timestamp"]
                })


    # 將匹配好的資料傳遞給模板
    return render_template('dataCollectionNew.html',matched_data=matched_data)

# -------------- new verson ---------------------
@app.route('/tongue_identify_2o')
def tongue_identify_2o():
    return render_template('tongue_identify_2o.html')


# 中醫師端頁面 New
@app.route('/doctorSide_2o')
def doctorSide_2o():
    json_path_tongue = "./static/dist/json/tongue_images_list.json"
    json_path_docSide = "./static/dist/json/doctorSide_list.json"
    
    try:
        # 讀取第一個 JSON 檔案
        with open(json_path_tongue, 'r') as json_file_tongue:
            tongue_data = json.load(json_file_tongue)
    except FileNotFoundError:
        tongue_data = []
    except json.JSONDecodeError:
        tongue_data = []
    except Exception as e:
        print(f"Error reading JSON file (tongue): {e}")
        tongue_data = []

    if tongue_data:
        tongue_images = tongue_data[0]
    else:
        tongue_images = {"tongue_original_img": [], "tongue_yolo_img": []}

    try:
        # 讀取第二個 JSON 檔案
        with open(json_path_docSide, 'r') as json_file_docSide:
            docSide_data = json.load(json_file_docSide)
    except FileNotFoundError:
        docSide_data = []
    except json.JSONDecodeError:
        docSide_data = []
    except Exception as e:
        print(f"Error reading JSON file (docSide): {e}")
        docSide_data = []

    if not docSide_data:
        docSide_data = {"some_default_key": "default_value"}

    # 將兩個 JSON 資料一起傳遞到模板
    return render_template('doctorSide_2o.html', tongue_images=tongue_images, docSide_data=docSide_data)


# # 中醫師端頁面 old
# @app.route('/doctorSide_2o')
# def doctorSide_2o():
#     json_path = "./static/dist/json/tongue_images_list.json"
#     json_path_docSide = "./static/dist/json/doctorSide_list.json"
    
#     try:
#         with open(json_path, 'r') as json_file:
#             data = json.load(json_file)
#     except FileNotFoundError:
#         data = []
#     except json.JSONDecodeError:
#         data = []
#     except Exception as e:
#         print(f"Error reading JSON file: {e}")
#         data = []

#     if data:
#         tongue_images = data[0]
#     else:
#         tongue_images = {"tongue_original_img": [], "tongue_yolo_img": []}

#     return render_template('doctorSide_2o.html', tongue_images=tongue_images)

# tongue_identify_3o 頁面
@app.route('/tongue_identify_3o')
def tongue_identify_3o():
    return render_template('tongue_identify_3o.html')

# result 舌象
@app.route('/ydts.html')
def ydts():
    return render_template('ydts.html')

@app.route('/qdbt.html')
def qdbt():
    return render_template('qdbt.html')

@app.route('/dhbt.html')
def dhbt():
    return render_template('dhbt.html')

@app.route('/yindts.html')
def yindts():
    return render_template('yindts.html')



if __name__ == '__main__':
    app.run(debug=True)
