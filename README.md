# 🫁 AI Traditional Chinese Medicine Tongue Diagnosis System
中醫 AI 舌診輔助系統

## 📌 Overview
本系統結合 YOLOv8 即時辨識技術與問診流程，
協助使用者初步了解自身健康狀況，
並提供中醫師診斷時的輔助參考資料。

## 🛠️ Tech Stack
- Python
- YOLOv8 + Roboflow（影像標記與模型訓練）
- Flask（Web 後端）
- 雲端資料庫（問診記錄儲存）

## 🧩 System Modules

### 1. AI 辨識模組
- 使用者對著鏡頭伸出舌頭，系統自動擷取舌頭特徵
- YOLOv8 模型進行即時舌診辨識
- 輸出辨識結果、症狀、健康狀態、建議與信心指數

### 2. 中醫師端模組
- 查看待協助案例與 AI 辨識圖片
- 填寫診斷表單並提交佐證資料
- 管理已協助案例紀錄

### 3. 辨識可靠度模組
- 比對 AI 辨識結果與中醫師佐證資料
- 圖表呈現辨識可靠度分析

## 🏗️ System Architecture
```
使用者問診 → YOLOv8 舌診辨識 → 舌診報告生成
                                      ↓
                              雲端資料庫儲存
                                      ↓
                          中醫師端審閱 → 可靠度比對分析
```

## 🚀 How to Run
```bash
pip install -r requirement.txt
python app.py
```
