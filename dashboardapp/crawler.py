# stockcrawler/crawler.py

import requests
import time
import json
from datetime import datetime, timedelta
import csv
import yfinance as yf
import pandas as pd
import os
import random

headers = {
    # Judy's User-Agent
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
}
def save_today_news(news_list, latest_date, keyword):
    today_file = 'today.csv'
    news_saved = False  # 紀錄是否有新聞存入 today.csv
    
    for news in news_list:
        title = news['title']
        if keyword in title:
            timestamp = news['publishAt']
            date_time = datetime.fromtimestamp(timestamp)
            date = date_time.strftime('%Y-%m-%d')

            # 如果新聞的日期等於最新日期，則儲存到 today.csv
            if date == latest_date:
                # 檢查是否已經存在 today.csv 檔案
                file_exists = os.path.isfile(today_file)
                with open(today_file, 'a', newline='', encoding='utf8') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    if not file_exists:
                        writer.writerow(['date', 'title'])  # 寫入標題
                    writer.writerow([date, title])
                    news_saved = True
                print(f'新增今日新聞: {title}')

    if not news_saved:
        print(f'當天 {latest_date} 沒有新聞')

def savefile(beginday, stopday, news):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(current_directory, 'cnyes-' + beginday + '~' + stopday + '.csv')
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if not file_exists:
            writer.writerow(['date', 'title'])
        writer.writerow(news)


# def save_news(news_list, beginday, stopday, keyword):
#     for news in news_list:
#         title = news['title']
#         if keyword in title:
#             timestamp = news['publishAt']
#             date_time = datetime.fromtimestamp(timestamp)

#             # 检查新闻是否在9:00之前发布
#             if date_time.hour < 9:
#                 date_time -= timedelta(days=1)

#             date = date_time.strftime('%Y-%m-%d')
#             news_data = [date, title]
#             savefile(beginday, stopday, news_data)
#             print(f'新增新聞: {title}')

def save_news(news_list, beginday, stopday, keyword, latest_date=None):
    for news in news_list:
        title = news['title']
        if keyword in title:
            timestamp = news['publishAt']
            date_time = datetime.fromtimestamp(timestamp)

            # 檢查新聞是否在9:00之前發布
            if date_time.hour < 9:
                date_time -= timedelta(days=1)

            date = date_time.strftime('%Y-%m-%d')
            news_data = [date, title]

            # 如果有傳遞 latest_date 參數，表示這是最後一天的新聞處理
            if latest_date and date == latest_date:
                save_today_news(news_list, latest_date, keyword)
            else:
                savefile(beginday, stopday, news_data)
                print(f'新增新聞: {title}')


# def crawler(request, beginday, stopday, keyword,max_retries=10):
#     # 將 beginday 和 stopday 轉換為 Timestamp
#     start_date = pd.Timestamp(beginday)
#     end_date = pd.Timestamp(stopday)

#      # 計算 startday 和 endday
#     startday = int(datetime.timestamp(start_date))
#     endday = int(datetime.timestamp(end_date) - 1)

#     # 爬取資料
#     # url = f'https://news.cnyes.com/api/v3/news/category/tw_stock?startAt={beginday}&endAt={stopday}&limit=30'
#     url = f'https://news.cnyes.com/api/v3/news/category/tw_stock?startAt={startday}&endAt={endday}&limit=30'
    
#     '''
#     res = requests.get(url, headers=headers)
#     data = json.loads(res.text)
#     last_page = data['items']['last_page']
#     print('總共 {} 頁'.format(last_page))

#     for page in range(1, last_page + 1):
#         print(f'正在爬取第 {page} 頁...')
#         page_url = f"{url}&page={page}"
#         res = requests.get(page_url, headers=headers)
#         news_list = json.loads(res.text)['items']['data']
#         save_news(news_list, beginday, stopday, keyword)
#         time.sleep(1)
#     '''
#     for attempt in range(max_retries):
#         try:
#             res = requests.get(url, headers=headers, timeout=10)
#             res.raise_for_status()  # 检查请求是否成功
#             data = json.loads(res.text)
#             last_page = data['items']['last_page']
#             print(f'總共 {last_page} 頁')

#             for page in range(1, last_page + 1):
#                 page_url = f"{url}&page={page}"
#                 for attempt_page in range(max_retries):
#                     try:
#                         res = requests.get(page_url, headers=headers, timeout=10)
#                         res.raise_for_status()
#                         news_list = json.loads(res.text)['items']['data']
#                         save_news(news_list, beginday, stopday, keyword)
#                         time.sleep(random.uniform(1, 3))  # 每次爬取后随机等待
#                         break
#                     except (requests.RequestException, json.JSONDecodeError) as e:
#                         print(f"第 {page} 頁出錯，重試 ({attempt_page + 1}/{max_retries})，錯誤: {e}")
#                         time.sleep(5)
#             break
#         except (requests.RequestException, json.JSONDecodeError) as e:
#             print(f"初始请求失敗，重試 ({attempt + 1}/{max_retries})，錯誤: {e}")
#             time.sleep(10)

def crawler(request, beginday, stopday, keyword, max_retries=10, latest_date=None):
    start_date = pd.Timestamp(beginday)
    end_date = pd.Timestamp(stopday)

    startday = int(datetime.timestamp(start_date))
    endday = int(datetime.timestamp(end_date) - 1)

    url = f'https://news.cnyes.com/api/v3/news/category/tw_stock?startAt={startday}&endAt={endday}&limit=30'

    for attempt in range(max_retries):
        try:
            res = requests.get(url, headers=headers, timeout=10)
            res.raise_for_status()
            data = json.loads(res.text)
            last_page = data['items']['last_page']
            print(f'總共 {last_page} 頁')

            for page in range(1, last_page + 1):
                page_url = f"{url}&page={page}"
                for attempt_page in range(max_retries):
                    try:
                        res = requests.get(page_url, headers=headers, timeout=10)
                        res.raise_for_status()
                        news_list = json.loads(res.text)['items']['data']

                        # 如果傳遞 latest_date 參數，表示這是最新一天的新聞處理
                        save_news(news_list, beginday, stopday, keyword, latest_date=latest_date)
                        time.sleep(random.uniform(1, 3))
                        break
                    except (requests.RequestException, json.JSONDecodeError) as e:
                        print(f"第 {page} 頁出錯，重試 ({attempt_page + 1}/{max_retries})，錯誤: {e}")
                        time.sleep(5)
            break
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"初始请求失敗，重試 ({attempt + 1}/{max_retries})，錯誤: {e}")
            time.sleep(10)




# def get_stock_data(stock, start_date, end_date):
#     ticker = yf.Ticker(stock)
#     df = ticker.history(start=start_date, end=end_date)
#     df.index = df.index.date

#     df['Next Close'] = df['Open'].shift(-1)
#     df['Price Difference'] = df['Next Close'] - df['Open'] 
#     df['next_ans'] = df['Price Difference'].apply(lambda x: 1 if x > 0 else 0)

#     df['avg3_ans'] = df['Open'].rolling(3).mean().shift(-3) - df['Open'].rolling(3).mean()
#     df['avg3_ans'] = df['avg3_ans'].apply(lambda x: 1 if x > 0 else 0)

#     df['avg5_ans'] = df['Open'].rolling(5).mean().shift(-5) - df['Open'].rolling(5).mean()
#     df['avg5_ans'] = df['avg5_ans'].apply(lambda x: 1 if x > 0 else 0)

#     df.reset_index(inplace=True)
#     df.rename(columns={'index': 'date'}, inplace=True)

#     try:
#         df = df.drop(df.index[:min(5, len(df))])
#         df = df.drop(df.index[-min(4, len(df)):])
#     except IndexError:
#         pass

#     df = df[['date', 'Open', 'next_ans', 'avg3_ans', 'avg5_ans']]
#     current_directory = os.path.dirname(os.path.abspath(__file__))
#     filename = os.path.join(current_directory, 'data.csv')
#     df.to_csv(filename, index=False)


def get_stock_data(stock, start_date, end_date):
    ticker = yf.Ticker(stock)
    df = ticker.history(start=start_date, end=end_date)
    df.index = df.index.date

    df['Next Close'] = df['Open'].shift(-1)
    df['Price Difference'] = df['Next Close'] - df['Open'] 
    df['next_ans'] = df['Price Difference'].apply(lambda x: 1 if x > 0 else 0)

    df['avg3_ans'] = df['Open'].rolling(3).mean().shift(-3) - df['Open'].rolling(3).mean()
    df['avg3_ans'] = df['avg3_ans'].apply(lambda x: 1 if x > 0 else 0)

    df['avg5_ans'] = df['Open'].rolling(5).mean().shift(-5) - df['Open'].rolling(5).mean()
    df['avg5_ans'] = df['avg5_ans'].apply(lambda x: 1 if x > 0 else 0)

    # 計算10日移動平均（MA）
    df['10_day_MA'] = df['Open'].rolling(window=10).mean()

    # 設定閾值 1% (0.01)
    threshold = 0.01

    # 初始化Signal列
    df['Signal'] = 0

    # 當開盤價低於10日均價且差距大於閾值，設置買入信號
    df.loc[(df['Open'] < df['10_day_MA']) & 
           ((df['10_day_MA'] - df['Open']) / df['10_day_MA'] > threshold), 'Signal'] = 1

    # 當開盤價高於10日均價，設置賣出信號
    df.loc[df['Open'] > df['10_day_MA'], 'Signal'] = -1

    # 重設索引並移除不需要的行
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)

    try:
        df = df.drop(df.index[:min(10, len(df))])
        # df = df.drop(df.index[-min(4, len(df)):])
    except IndexError:
        pass

    # 選擇需要的欄位
    df = df[['date', 'Open', 'next_ans', 'avg3_ans', 'avg5_ans', '10_day_MA', 'Signal']]

    # 匯出到csv檔案
    current_directory = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(current_directory, 'data.csv')
    df.to_csv(filename, index=False)

'''
def run_crawler(request, beginyear, beginmonth, stopmonth=12, keywords=None, stocks=None, start_date=None, end_date=None):
    # 如果 start_date 和 end_date 是字串，則轉換為 datetime 物件
    if isinstance(start_date, str):
        start_date = pd.Timestamp(start_date)
    if isinstance(end_date, str):
        end_date = pd.Timestamp(end_date)

    # 將 Timestamp 轉換為字串
    start_date_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
    end_date_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

    # 使用 strptime 轉換為 datetime 物件
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d %H:%M:%S")

    # 計算月份差
    num_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1

    print(f'計算出的 num_months 為: {num_months}')
    print(f'計算出的 start-dt 為: {start_date}')
    print(f'計算出的 year 為: {end_dt.year - start_dt.year}')
    print(f'計算出的 month 為: {(end_dt.month - start_dt.month)}')
    print(f'計算出的 em 為: {end_dt.month}')
    print(f'計算出的 sm 為: {(start_dt.month)}')
    print(f'計算出的 end-dt 為: {end_date}')

    # 將月份數存入 session
    request.session['num_months'] = num_months
    request.session['startday'] = start_date_str
    request.session['endday'] = end_date_str
    
    if keywords:
        for keyword in keywords:
            for m in range(beginmonth, stopmonth + 1):
                beginday = f'{beginyear}-{m:02d}-01'
                if m == 12:
                    stopday = f'{beginyear + 1}-01-01'
                else:
                    stopday = f'{beginyear}-{m + 1:02d}-01'
                crawler(request, beginday, stopday, keyword)
                time.sleep(5)

    if stocks:
        for stock in stocks:
            get_stock_data(stock, start_date, end_date)
            time.sleep(5)

'''
# def run_crawler(request, beginyear, beginmonth, stopmonth=12, keywords=None, stocks=None, start_date=None, end_date=None):
#     # 如果 start_date 和 end_date 是字串，則轉換為 datetime 物件
#     if isinstance(start_date, str):
#         start_date = pd.Timestamp(start_date)
#     if isinstance(end_date, str):
#         end_date = pd.Timestamp(end_date)

#     # 使用 strptime 轉換為 datetime 物件
#     start_dt = start_date.to_pydatetime()
#     end_dt = end_date.to_pydatetime()

#     # 計算月份差
#     num_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1

#     print(f'計算出的 num_months 為: {num_months}')
#     print(f'計算出的 start-dt 為: {start_date}')
#     print(f'計算出的 end-dt 為: {end_date}')

#     # 將月份數存入 session
#     request.session['num_months'] = num_months
#     request.session['startday'] = start_date.strftime("%Y-%m-%d")
#     request.session['endday'] = end_date.strftime("%Y-%m-%d")

#     if keywords:
#         current_year = start_dt.year
#         current_month = start_dt.month

#         # 遍历每个月
#         for i in range(num_months):
#             beginday = f'{current_year}-{current_month:02d}-01'
#             if current_month == 12:
#                 stopday = f'{current_year + 1}-01-01'
#                 current_year += 1
#                 current_month = 1
#             else:
#                 stopday = f'{current_year}-{current_month + 1:02d}-01'
#                 current_month += 1

#             for keyword in keywords:
#                 crawler(request, beginday, stopday, keyword)
#                 time.sleep(5)

#     if stocks:
#         for stock in stocks:
#             get_stock_data(stock, start_date, end_date)
#             time.sleep(5)

def run_crawler(request, beginyear, beginmonth, stopmonth=12, keywords=None, stocks=None, start_date=None, end_date=None):
    if isinstance(start_date, str):
        start_date = pd.Timestamp(start_date)
    if isinstance(end_date, str):
        end_date = pd.Timestamp(end_date)

    start_dt = start_date.to_pydatetime()
    end_dt = end_date.to_pydatetime()

    num_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1

    print(f'計算出的 num_months 為: {num_months}')
    print(f'計算出的 start-dt 為: {start_date}')
    print(f'計算出的 end-dt 為: {end_date}')

    request.session['num_months'] = num_months
    request.session['startday'] = start_date.strftime("%Y-%m-%d")
    request.session['endday'] = end_date.strftime("%Y-%m-%d")

    latest_date = end_date.strftime('%Y-%m-%d')  # 取得最新一天的日期
    latest_date = end_date  - timedelta(days=1)
    latest_date = latest_date.strftime('%Y-%m-%d')

    if keywords:
        current_year = start_dt.year
        current_month = start_dt.month

        # 遍歷每個月
        for i in range(num_months):
            beginday = f'{current_year}-{current_month:02d}-01'
            if current_month == 12:
                stopday = f'{current_year + 1}-01-01'
                current_year += 1
                current_month = 1
            else:
                stopday = f'{current_year}-{current_month + 1:02d}-01'
                current_month += 1

            for keyword in keywords:
                # 當月份為最後一個月份時，傳入 latest_date
                if i == num_months - 1:
                    crawler(request, beginday, stopday, keyword, latest_date=latest_date)
                else:
                    crawler(request, beginday, stopday, keyword)
                time.sleep(5)

    if stocks:
        for stock in stocks:
            get_stock_data(stock, start_date, end_date)
            time.sleep(5)



def merge_csv_files():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    cnyes_files = [f for f in os.listdir(current_directory) if f.startswith("cnyes") and f.endswith(".csv")]
    data_file = [f for f in os.listdir(current_directory) if f.startswith("data") and f.endswith(".csv")]

    if len(data_file) != 1:
        raise FileNotFoundError("data檔案不唯一！！")
    
    data_df = pd.read_csv(os.path.join(current_directory, data_file[0]))
    combined_df = pd.DataFrame()

    for file in cnyes_files:
        file = pd.read_csv(os.path.join(current_directory, file))
        combined_df = pd.concat([combined_df, file], ignore_index=True)

    merged_df = pd.merge(combined_df, data_df, on='date', how='inner')
    output_file = os.path.join(current_directory, "all.csv")
    merged_df.to_csv(output_file, index=False)

    print(f"合併的檔案已保存為 {output_file}")
