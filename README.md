# SmartStock Forecast 📈

利用新聞標題進行股價漲跌預測的 NLP 系統  
A Django-based NLP platform for forecasting stock price trend and simulating automated investment strategies.

👉 [點我觀看首頁介面影片](https://drive.google.com/file/d/1znLUg5sbLUwWPSjtNGznw1GR07FSxsUM/view?usp=drive_link)


---

## 📌 專題簡介

本研究開發一套**結合自然語言處理 (NLP)** 與 **股市資料分析**的預測系統，透過分析新聞標題，預測四家台灣電子股（鴻海、台積電、聯發科、日月光）的漲跌趨勢，並設計**一鍵式投資策略模擬平台**。使用者只需輸入股票代碼，即可自動完成爬蟲、預測、回測並產出報表。

**研究成果包含：**
- 📊 NLP 預測模型準確度達 74%
- 💰 實驗組策略平均年投報率高達 38.70%
- 🧠 採用 CKIP BERT + 滾動切分 + 回測模擬
- 🖥️ 前後端整合 Django 操作介面

---

## 🧠 系統架構

> 你可以貼一張來自海報或報告的系統流程圖  
> 或使用簡單的步驟文字敘述：

1. 使用者輸入目標股票（名稱與代碼）
2. 系統執行爬蟲抓取新聞與股價
3. 自動進行 NLP 標註 → CKIP BERT 預測漲跌
4. 將預測結果加入技術分析條件，回測投資策略
5. 產出圖表與結果

---

## 🖥️ 安裝與執行方式

```bash
# 1. 複製此專案
git clone https://github.com/Tsaili-17/SmartStock-Forecast.git
cd SmartStock-Forecast

# 2. 安裝虛擬環境（可選）
python -m venv venv
venv\Scripts\activate

# 3. 安裝必要套件
pip install -r requirements.txt

# 4. 執行 Django
python manage.py runserver
