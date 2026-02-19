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
