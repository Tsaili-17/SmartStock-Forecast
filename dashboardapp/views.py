# 基本库
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import DateOffset

from ckiptagger import WS

# PyTorch 和 transformers
import torch
from torch.nn import Module, Linear, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizerFast, AutoModel
from tqdm import tqdm

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.utils import resample

# 数据可视化
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.figure_factory as ff
import plotly.express as px

# Django 框架
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings

# 项目中的模型、表单和爬虫工具
from .models import My_Dash, data_decoded
from .forms import StockCrawlForm
from .crawler import run_crawler, merge_csv_files


# Create your views here.

dash = My_Dash()
    
def home(request):
    return render(request, "page0_home.html")

def page1_upload(request):
    global dash
    if request.method == 'POST':
        form = StockCrawlForm(request.POST)
        if form.is_valid():
            # 获取用户输入的数据
            
            keyword = form.cleaned_data['keyword']
            stock_codes = form.cleaned_data['stock_codes'].split(',')
            # end_date = datetime.today()
            end_date = '2024-10-31'
            # 將字串轉換為 datetime 物件
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            start_date = (end_date - relativedelta(years=2, months=10)).replace(day=1)

            # 清理股票代码，并确保每个代码加上 .TW
            stock_codes = [code.strip() + '.TW' for code in stock_codes]

            # 调用爬虫函数
            beginday = start_date.strftime('%Y-%m-%d')
            stopday = end_date.strftime('%Y-%m-%d')
            print(f'計算出的 start-dt 為: {beginday}')
            print(f'計算出的 end-dt 為: {stopday}')
            # 修改這裡，將 request 參數傳遞給 run_crawler
            run_crawler(
                request,  # 新增這一行，傳遞 request
                beginyear=start_date.year,
                beginmonth=start_date.month,
                stopmonth=end_date.month,
                keywords=[keyword],  # 关键词列表
                stocks=stock_codes,  # 股票代码列表
                start_date=beginday,  # 开始日期
                end_date=stopday      # 结束日期
            )
            # 呼叫函式進行合併
            merge_csv_files()
            #messages.success(request, 'Data crawled and saved successfully!')
            return render(request, 'page1_1_visualization.html', {'form': form, 'success': True})
    else:
        form = StockCrawlForm()
        #messages.warning(request, 'Send the form correctly!!')
    return render(request, 'page1_crawler.html', {'form': form})

def page2_datatable(request):
    global dash

    # 獲取目前腳本檔案的目錄
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # 定義"all.csv"的路徑
    file_path = os.path.join(current_directory, 'all.csv')

    # 检查文件是否存在
    if os.path.exists(file_path) and request.method == 'POST' and 'data_upload_button' in request.POST:
    # if os.path.exists(file_path):
        try:
            # 读取 "all.csv"
            dash.df1 = pd.read_csv(file_path)

            if dash.df1 is not None and not dash.df1.empty:
                target_columns = ['next_ans', 'avg3_ans', 'avg5_ans']
                dash.slt_feat = ['date', 'title']

                # Store dataframe and target column options in session
                request.session['data'] = dash.df1.to_dict(orient='records')
                request.session['target_options'] = target_columns
                next_ans=['next_ans']
                context = {
                    'train_data': dash.df1.values.tolist(),
                    'train_columns': dash.df1.columns,
                    'target_options': next_ans,
                    'selected_flag': 'False'
                }
                return render(request, 'page2_datatable.html', context)
            else:
                messages.warning(request, 'The file "all.csv" is empty or failed to load.')
                return render(request, 'page1_crawler.html')  # 返回到文件上传页面
        except Exception as e:
            messages.error(request, f'Error loading "all.csv": {e}')
            return render(request, 'page1_crawler.html')  # 返回到文件上传页面
    
    elif request.method == 'POST' and 'target_confirm_button' in request.POST:
        if dash.df1 is not None and not dash.df1.empty:
            dash.target_col = request.POST.get('target_slt')

            if dash.target_col in dash.slt_feat:
                dash.slt_feat.remove(dash.target_col)
                
            # Store target column in session
            request.session['target_slt'] = dash.target_col

            context = {
                'train_data': dash.df1.values.tolist(),
                'train_columns': dash.df1.columns,
                'target_slt': dash.target_col,
                'selected_flag': 'True'
            }

            return render(request, 'page3_train_test_comparison.html', context)
        else:
            messages.warning(request, 'No data available. Please crawl data first.')
            return render(request, 'page1_1_visualization.html')  # 返回到文件上传页面

    else:
        messages.warning(request, 'You have to crawl data first!')
        return render(request, 'page1_1_visualization.html')  # 返回到文件上传页面
    

# 設置隨機種子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多個GPU

# 定義BERT模型和分類頭
class BertForClassification(Module):
    def __init__(self, bert_model, num_labels):
        super(BertForClassification, self).__init__()
        self.bert = bert_model
        self.classifier = Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # 取 BERT 最後一層的[CLS]標誌的輸出
        logits = self.classifier(pooled_output)
        return logits

# 函數：準備數據為 BERT 所需格式
def prepare_test_data(data, is_training=True):
    # Tokenize the text data
    inputs = tokenizer(list(data['title']), return_tensors='pt', padding=True, truncation=True, max_length=512)
    # Create a dataset and dataloader
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    return DataLoader(dataset, batch_size=64, shuffle=is_training)  # Shuffle only during training

# 函數：準備數據為 BERT 所需格式
def prepare_data(data, is_training=True):
    # Tokenize the text data
    inputs = tokenizer(list(data['title']), return_tensors='pt', padding=True, truncation=True, max_length=512)
    labels = torch.tensor(data['next_ans'].values)
    # Create a dataset and dataloader
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
    return DataLoader(dataset, batch_size=64, shuffle=is_training)  # Shuffle only during training

# 函數3：準備數據為 BERT 所需格式
def prepare_data3(data, is_training=True):
    # Tokenize the text data
    inputs = tokenizer(list(data['title']), return_tensors='pt', padding=True, truncation=True, max_length=512)
    labels = torch.tensor(data['avg3_ans'].values)
    # Create a dataset and dataloader
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
    return DataLoader(dataset, batch_size=64, shuffle=is_training)  # Shuffle only during training

# 函數5：準備數據為 BERT 所需格式
def prepare_data5(data, is_training=True):
    # Tokenize the text data
    inputs = tokenizer(list(data['title']), return_tensors='pt', padding=True, truncation=True, max_length=512)
    labels = torch.tensor(data['avg5_ans'].values)
    # Create a dataset and dataloader
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
    return DataLoader(dataset, batch_size=64, shuffle=is_training)  # Shuffle only during training

# 函數：訓練模型
def train_model(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 函數：驗證或測試模型
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_preds

# 函數：驗證或測試模型
def test_evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    # all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            # all_labels.extend(labels.cpu().numpy())
    # accuracy = accuracy_score(all_labels, all_preds)
    return  all_preds

# 設置 BERT 模型與 tokenizer
model_name = 'ckiplab/bert-base-chinese'
tokenizer = BertTokenizerFast.from_pretrained(model_name)

# 設置設備：確認使用 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Django 的 view 方法整合：
def page3_train_test_comparison(request):
    global dash
    if request.method == 'POST':
        # 获取用户选择的切分方法
        split_method = request.POST.get('split_method')
        
        # 从 session 中获取数据和 target 列
        df_data = request.session.get('data')
        target_col = request.session.get('target_slt')

        target_col = 'next_ans'  # 這裡要改成 'avg3_ans' 或 'avg5_ans'

        if df_data is None or target_col is None:
            # 数据或 target 列未找到
            messages.warning(request, 'Data or target column not found. Please upload and select data first.')
            return redirect('train_test_comparison')  # 根据实际 URL 修改
        
        # 将 session 中的字典数据转换为 DataFrame
        df = pd.DataFrame(df_data)
        
        # 確認數據格式符合需求
        data = df
        data['date'] = pd.to_datetime(data['date'])

        if split_method == 'uniform_split':
            train_data, test_data = train_test_split(
                data, 
                test_size=0.1, 
                stratify=data[target_col], 
                random_state=42
            )

            # 將 date 欄位轉換為字串格式
            train_data['date'] = train_data['date'].astype(str)
            test_data['date'] = test_data['date'].astype(str)

            # 然後再存入 session 中
            request.session['train_data'] = train_data.to_dict(orient='records')
            request.session['test_data'] = test_data.to_dict(orient='records')

            
            # 將結果存儲到 session 中
            request.session['train_data'] = train_data.to_dict(orient='records')
            request.session['test_data'] = test_data.to_dict(orient='records')

            # 初始化顯示的數據集為訓練集
            current_data = 'train'
            selected_data = train_data.to_dict(orient='records')
            columns = train_data.columns.tolist()

            return render(request, 'page3_train_test_comparison.html', {
                'selected_data': selected_data,
                'current_data': current_data,
                'columns': columns
            })
        
        
        elif split_method == 'rolling_split':
            # 在選擇 rolling_split 時，不需要返回顯示訓練集和測試集，執行滾動切分邏輯
            print(data.columns)  # 檢查欄位名稱是否有 'avg3_ans'
            # print(data.head())   # 檢查前幾筆資料，確認 'avg3_ans' 的數值是否存在

            # 設置 BERT 模型與 tokenizer
            model_name = 'ckiplab/bert-base-chinese'
            tokenizer = BertTokenizerFast.from_pretrained(model_name)
            bert_model = AutoModel.from_pretrained(model_name)

            # 在初始化模型之前設置隨機種子
            seed = 42
            set_seed(seed)

            # 初始化模型、優化器和損失函數
            model = BertForClassification(bert_model, num_labels=2)
            optimizer = Adam(model.parameters(), lr=9e-6)
            criterion = CrossEntropyLoss()

            # 設備設置
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(torch.cuda.is_available())
            model.to(device)
            num_months = request.session.get('num_months', None)
            startday = request.session.get('startday', None)
            endday = request.session.get('endday', None)

            # 滾動訓練與測試邏輯
            output = pd.DataFrame(columns=['date', 'title', 'predicted_next_ans', 'actual_next_ans'])
            # # 使用 pandas 將 UNIX 時間戳轉換為日期
            # startday_date = pd.to_datetime(startday, unit='s')

            # 轉換為你需要的格式
            startday = datetime.strptime(startday, "%Y-%m-%d")
            # startday = datetime.strptime(startday, "%Y-%m-%d %H:%M:%S")
            start_date = pd.Timestamp(startday)
            # start_date = pd.Timestamp('2024-01-01')
            all_test_preds = []
            all_test_labels = []
            print(num_months)

            # for i in range(num_months-1-13):  # 滾動 ? 次來涵蓋所有時間範圍
            for i in range(num_months-13):    
                train_start = start_date + DateOffset(months=i)
                train_end = train_start + DateOffset(months=12) ###train 的時長改這裡(next_ans)
                val_end = train_end + DateOffset(months=1)
                test_end = val_end + DateOffset(months=1)

                train_data = data[(data['date'] >= train_start) & (data['date'] < train_end)]
                val_data = data[(data['date'] >= train_end) & (data['date'] < val_end)]
                test_data = data[(data['date'] >= val_end) & (data['date'] < test_end)]

                train_loader = prepare_data(train_data, is_training=True)
                val_loader = prepare_data(val_data, is_training=False)
                test_loader = prepare_data(test_data, is_training=False)

                print(f"\nTraining from {train_start} to {train_end}")
                train_loss = train_model(model, train_loader, optimizer, criterion)

                print(f"Validating from {train_end} to {val_end}")
                val_accuracy, _ = evaluate_model(model, val_loader)
                print(f"Validation Accuracy: {val_accuracy:.4f}")

                print(f"Testing from {val_end} to {test_end}")
                test_accuracy, test_preds = evaluate_model(model, test_loader)
                print(f"Test Accuracy: {test_accuracy:.4f}")

                all_test_preds.extend(test_preds)
                all_test_labels.extend(test_data['next_ans'].values)

                # 保存測試結果
                output = pd.concat([output, pd.DataFrame({
                    'date': test_data['date'].values,
                    'title': test_data['title'].values,
                    'predicted_next_ans': test_preds,
                    'actual_next_ans': test_data['next_ans'].values
                })], ignore_index=True)

            # 保存模型
            model_save_path = "trained_model.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"模型已保存至 {model_save_path}")


            # 找到最後一天的新聞資料與預測結果
            last_test_date = test_data['date'].max()
            last_day_data = output[output['date'] == last_test_date].copy()

            # 設置預設權重，避免 SettingWithCopyWarning
            last_day_data.loc[:, 'weight'] = 1

            # 計算該日期所有 'predicted_next_ans' 的加權平均
            if last_day_data['weight'].sum() == 0:
                weighted_avg = 0
            else:
                weighted_avg = (last_day_data['predicted_next_ans'] * last_day_data['weight']).sum() / last_day_data['weight'].sum()

            # 判斷加權平均是否大於等於 0.5 並設定結果
            final_prediction = 1 if weighted_avg >= 0.5 else 0

            # 存入 show DataFrame 並輸出結果
            show = pd.DataFrame({
                'date': [last_test_date],
                'weighted_average': [weighted_avg],
                'final_prediction': [final_prediction]
            })
            prediction_text = 'Rise' if final_prediction == 1 else 'Fall'

            # 這裡要做avg3-ans的滾動結果
            model = BertForClassification(bert_model, num_labels=2)
            optimizer = Adam(model.parameters(), lr=9e-6)
            criterion = CrossEntropyLoss()

            # 設備設置
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            num_months = request.session.get('num_months', None)
            startday = request.session.get('startday', None)
            endday = request.session.get('endday', None)

            # 滾動訓練與測試邏輯
            output3 = pd.DataFrame(columns=['date', 'title', 'predicted_avg3_ans', 'actual_avg3_ans'])
            # # 使用 pandas 將 UNIX 時間戳轉換為日期
            # startday_date = pd.to_datetime(startday, unit='s')

            # 轉換為你需要的格式
            startday = datetime.strptime(startday, "%Y-%m-%d")
            # startday = datetime.strptime(startday, "%Y-%m-%d %H:%M:%S")
            start_date = pd.Timestamp(startday)
            # start_date = pd.Timestamp('2024-01-01')
            all_test_preds = []
            all_test_labels = []
            print(num_months)

            # for i in range(num_months-1-13):
            for i in range(num_months-13):  # 滾動 ? 次來涵蓋所有時間範圍
                train_start = start_date + DateOffset(months=i)
                train_end = train_start + DateOffset(months=12) ###train 的時長改這裡(avg3_ans)
                val_end = train_end + DateOffset(months=1)
                test_end = val_end + DateOffset(months=1)

                train_data = data[(data['date'] >= train_start) & (data['date'] < train_end)]
                val_data = data[(data['date'] >= train_end) & (data['date'] < val_end)]
                test_data = data[(data['date'] >= val_end) & (data['date'] < test_end)]

                train_loader = prepare_data3(train_data, is_training=True)
                val_loader = prepare_data3(val_data, is_training=False)
                test_loader = prepare_data3(test_data, is_training=False)

                print(f"\nTraining from {train_start} to {train_end}")
                train_loss = train_model(model, train_loader, optimizer, criterion)

                print(f"Validating from {train_end} to {val_end}")
                val_accuracy, _ = evaluate_model(model, val_loader)
                print(f"Validation Accuracy: {val_accuracy:.4f}")

                print(f"Testing from {val_end} to {test_end}")
                test_accuracy, test_preds = evaluate_model(model, test_loader)
                print(f"Test Accuracy: {test_accuracy:.4f}")

                all_test_preds.extend(test_preds)
                all_test_labels.extend(test_data['avg3_ans'].values)

                # 保存測試結果
                output3 = pd.concat([output3, pd.DataFrame({
                    'date': test_data['date'].values,
                    'title': test_data['title'].values,
                    'predicted_avg3_ans': test_preds,
                    'actual_avg3_ans': test_data['avg3_ans'].values
                })], ignore_index=True)

            # 找到最後一天的新聞資料與預測結果
            last_test_date = test_data['date'].max()
            last_day_data = output3[output3['date'] == last_test_date].copy()

            # 設置預設權重，避免 SettingWithCopyWarning
            last_day_data.loc[:, 'weight'] = 1

            # 計算該日期所有 'predicted_next_ans' 的加權平均
            if last_day_data['weight'].sum() == 0:
                weighted_avg = 0
            else:
                weighted_avg = (last_day_data['predicted_avg3_ans'] * last_day_data['weight']).sum() / last_day_data['weight'].sum()

            # 判斷加權平均是否大於等於 0.5 並設定結果
            final_prediction3 = 1 if weighted_avg >= 0.5 else 0

            # 存入 show DataFrame 並輸出結果
            show = pd.DataFrame({
                'date': [last_test_date],
                'weighted_average': [weighted_avg],
                'final_prediction3': [final_prediction3]
            })
            prediction_text3 = 'Rise' if final_prediction3 == 1 else 'Fall'

            # 這裡要做avg5-ans的滾動結果
            model = BertForClassification(bert_model, num_labels=2)
            optimizer = Adam(model.parameters(), lr=9e-6)
            criterion = CrossEntropyLoss()

            # 設備設置
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            num_months = request.session.get('num_months', None)
            startday = request.session.get('startday', None)
            endday = request.session.get('endday', None)

            # 滾動訓練與測試邏輯
            output5 = pd.DataFrame(columns=['date', 'title', 'predicted_avg5_ans', 'actual_avg5_ans'])
            # # 使用 pandas 將 UNIX 時間戳轉換為日期
            # startday_date = pd.to_datetime(startday, unit='s')

            # 轉換為你需要的格式
            startday = datetime.strptime(startday, "%Y-%m-%d")
            # startday = datetime.strptime(startday, "%Y-%m-%d %H:%M:%S")
            start_date = pd.Timestamp(startday)
            # start_date = pd.Timestamp('2024-01-01')
            all_test_preds = []
            all_test_labels = []
            print(num_months)

            # for i in range(num_months-1-13):
            for i in range(num_months-13):  # 滾動 ? 次來涵蓋所有時間範圍
                train_start = start_date + DateOffset(months=i)
                train_end = train_start + DateOffset(months=12) ###train 的時長改這裡(avg5_ans)
                val_end = train_end + DateOffset(months=1)
                test_end = val_end + DateOffset(months=1)

                train_data = data[(data['date'] >= train_start) & (data['date'] < train_end)]
                val_data = data[(data['date'] >= train_end) & (data['date'] < val_end)]
                test_data = data[(data['date'] >= val_end) & (data['date'] < test_end)]

                train_loader = prepare_data5(train_data, is_training=True)
                val_loader = prepare_data5(val_data, is_training=False)
                test_loader = prepare_data5(test_data, is_training=False)

                print(f"\nTraining from {train_start} to {train_end}")
                train_loss = train_model(model, train_loader, optimizer, criterion)

                print(f"Validating from {train_end} to {val_end}")
                val_accuracy, _ = evaluate_model(model, val_loader)
                print(f"Validation Accuracy: {val_accuracy:.4f}")

                print(f"Testing from {val_end} to {test_end}")
                test_accuracy, test_preds = evaluate_model(model, test_loader)
                print(f"Test Accuracy: {test_accuracy:.4f}")

                all_test_preds.extend(test_preds)
                all_test_labels.extend(test_data['avg5_ans'].values)

                # 保存測試結果
                output5 = pd.concat([output5, pd.DataFrame({
                    'date': test_data['date'].values,
                    'title': test_data['title'].values,
                    'predicted_avg5_ans': test_preds,
                    'actual_avg5_ans': test_data['avg5_ans'].values
                })], ignore_index=True)

            # 找到最後一天的新聞資料與預測結果
            last_test_date = test_data['date'].max()
            last_day_data = output5[output5['date'] == last_test_date].copy()

            # 設置預設權重，避免 SettingWithCopyWarning
            last_day_data.loc[:, 'weight'] = 1

            # 計算該日期所有 'predicted_next_ans' 的加權平均
            if last_day_data['weight'].sum() == 0:
                weighted_avg = 0
            else:
                weighted_avg = (last_day_data['predicted_avg5_ans'] * last_day_data['weight']).sum() / last_day_data['weight'].sum()

            # 判斷加權平均是否大於等於 0.5 並設定結果
            final_prediction5 = 1 if weighted_avg >= 0.5 else 0

            # 存入 show DataFrame 並輸出結果
            show = pd.DataFrame({
                'date': [last_test_date],
                'weighted_average': [weighted_avg],
                'final_prediction5': [final_prediction5]
            })
            prediction_text5 = 'Rise' if final_prediction5 == 1 else 'Fall'

            # 定義模型加載函數
            def load_model(model, model_path):
                model.load_state_dict(torch.load(model_path))
                model.eval()  # 切換模型為推理模式
                return model

            # 檢查 today.csv 是否存在
            csv_file = "today.csv"
            if os.path.exists(csv_file):
                # 讀取 today.csv
                today_data = pd.read_csv(csv_file)
                
                # if 'title' not in today_data.columns or 'next_ans' not in today_data.columns:
                #     print("today.csv 中沒有 'title' 或 'next_ans' 欄位")
                if 'title' not in today_data.columns :
                    print("today.csv 中沒有 'title'")
                else:
                    # 加載已保存的模型
                    model = load_model(model, model_save_path)
                    
                    # 準備數據
                    today_loader = prepare_test_data(today_data, is_training=False)
                    
                    # 預測
                    predictions = test_evaluate_model(model, today_loader)
                    
                    # 計算平均結果
                    avg_prediction = sum(predictions) / len(predictions)
                    final_result = 1 if avg_prediction >= 0.5 else 0
                    
                    for idx, (title, prediction) in enumerate(zip(today_data['title'], predictions)):
                        trend = '漲' if prediction == 1 else '跌'
                        print(f"新聞 {idx+1}: {title}")
                        print(f"預測結果: {prediction} ({trend})\n")
                    
                    print(f"平均預測結果為: {avg_prediction:.2f}, 最終判斷為: {final_result}")
                    final_result = 'Rise' if final_result == 1 else 'Fall'
                    print(f"隔日漲還是跌: {final_result}")
            else:
                print("沒有找到 today.csv 文件，無法進行預測。")

            # 開始計算報酬率
            output['date'] = pd.to_datetime(output['date'])
            output.sort_values(by='date', inplace=True)

            result = output.groupby('date')['predicted_next_ans'].mean().reset_index()
            result['date'] = pd.to_datetime(result['date'])

            # 定義數據路徑
            current_directory = os.path.dirname(os.path.abspath(__file__))
            data_file_path = os.path.join(current_directory, 'data.csv')
            dash.df1 = pd.read_csv(data_file_path)

            dash.df1['date'] = pd.to_datetime(dash.df1['date'], errors='coerce')

            if 'date' in dash.df1.columns and 'Open' in dash.df1.columns:
                stock = dash.df1[['date', 'Open', 'Signal']]
                stock['date'] = pd.to_datetime(stock['date'], errors='coerce')
                merged_df = pd.merge(result, stock, on='date', how='right')
                # merged_df.ffill(inplace=True)
                # merged_df.bfill(inplace=True)
            else:
                print("Columns 'date' and 'Open' not found in dash.df1.")
                merged_df = pd.DataFrame()
            # print(merged_df.head())
            # 確認 result 中的最早日期
            earliest_date_in_result = result['date'].min()

            # 依據最早日期篩選 merged_df 中符合條件的資料
            # 保留 merged_df 中 `date` 欄位大於等於 `earliest_date_in_result` 的資料
            merged_df = merged_df.loc[merged_df['date'] >= earliest_date_in_result]
            print(merged_df.head())

           # 繪製月度回報的函數
            def plot_monthly_returns(daily_returns, predicted_returns, filename):
                images_dir = os.path.join('static', 'images')
                if not os.path.exists(images_dir):
                    os.makedirs(images_dir)
                
                chart_path = os.path.join(images_dir, filename)
                
                plt.plot(daily_returns['year_month'], daily_returns['return'], label='Strategy 1 (MA 10 Investment)', marker='o')
                plt.plot(predicted_returns['year_month'], predicted_returns['return'], label='Strategy 2 (MA 10 & Predictions Investment)', marker='o')
                plt.title('Comparison of Monthly Return: Strategy 1 vs. Strategy 2')
                plt.xlabel('Month') 
                plt.ylabel('Monthly Return')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(chart_path)
                plt.close()
                
                return chart_path

            def calculate_monthly_return(strategy):
                monthly_returns = []
                merged_df['year_month'] = merged_df['date'].dt.to_period('M')
                grouped = merged_df.groupby('year_month')

                for name, group in grouped:
                    invest_days = []
                    total_investment = 0  # 總投資次數（每次1000台幣）
                    total_profit = 0  # 總獲利
                    if strategy == "Signal":
                        # 策略一：根據 Signal 買賣
                        invest_days = group[group['Signal'] == 1].index
                        sell_days = group[group['Signal'] == -1].index

                        for buy_day in invest_days:
                            buy_price = group.loc[buy_day, 'Open']
                            sell_day = sell_days[sell_days > buy_day].min()  # 找到最近的賣出日

                            if pd.notna(sell_day):
                                sell_price = group.loc[sell_day, 'Open']
                                shares_bought = 1000 / buy_price
                                profit = (sell_price - buy_price) * shares_bought
                                total_profit += profit
                                total_investment += 1000
                    elif strategy == "Moving_Average_and_predicted":
                        # 策略四：基於10日移動平均線買入，10日移動平均為-1且預測為1時賣出
                        invest_days = group[group['Signal'] == 1].index

                        for buy_day in invest_days:
                            buy_price = group.loc[buy_day, 'Open']
                            sell_candidates =  group[(group['Signal'] == -1) & (group['predicted_next_ans'] == 0)]
                            sell_day = sell_candidates.index.min() if not sell_candidates.empty else None  # 找到最近的賣出日

                            if pd.notna(sell_day):
                                sell_price = group.loc[sell_day, 'Open']
                                shares_bought = 1000 / buy_price
                                profit = (sell_price - buy_price) * shares_bought
                                total_profit += profit
                                total_investment += 1000

                    if total_investment > 0:
                        monthly_return = total_profit / total_investment
                    else:
                        monthly_return = 0

                    monthly_returns.append({'year_month': str(name), 'return': monthly_return})

                return pd.DataFrame(monthly_returns)

            # 計算日常和預測策略的月度回報
            daily_returns = calculate_monthly_return(strategy="Signal")
            print(daily_returns)
            predicted_returns = calculate_monthly_return(strategy="Moving_Average_and_predicted")
            print(predicted_returns)
            # 保存圖像並獲取圖像路徑
            chart_path = plot_monthly_returns(daily_returns, predicted_returns, 'monthly_return_comparison.png')
            
            daily_profit_months = (daily_returns['return'] > 0).sum()
            predicted_profit_months = (predicted_returns['return'] > 0).sum()

            nonzero_signal_returns = daily_returns.loc[daily_returns['return'] != 0]
            nonzero_predicted_returns = predicted_returns.loc[predicted_returns['return'] != 0]
            daily_avg_return = nonzero_signal_returns['return'].mean()
            predicted_avg_return = nonzero_predicted_returns['return'].mean()

            daily_volatility = daily_returns['return'].std()
            predicted_volatility = predicted_returns['return'].std()

            def calculate_max_drawdown(returns):
                cumulative_returns = (1 + returns).cumprod()
                drawdown = cumulative_returns / cumulative_returns.cummax() - 1
                return drawdown.min()

            daily_max_drawdown = calculate_max_drawdown(daily_returns['return'])
            predicted_max_drawdown = calculate_max_drawdown(predicted_returns['return'])

            comparison = {
                'Strategy 1 (MA 10 Investmentt)': {
                    'Number_of_profitable_trades': daily_profit_months,
                    'Annual_return_rate': "{:.3f}".format(round(daily_avg_return * 12, 5)*100),
                    'Volatility_of_returns': "{:.3f}".format(round(daily_volatility, 5) * 100),
                    'Maximal_drawdown': daily_max_drawdown
                },
                'Strategy 2 (MA 10 & Predictions Investment)': {
                    'Number_of_profitable_trades': predicted_profit_months,
                    'Annual_return_rate': "{:.3f}".format(round(predicted_avg_return * 12*100, 3)),
                    'Volatility_of_returns': "{:.3f}".format(round(predicted_volatility, 5) * 100),
                    'Maximal_drawdown': predicted_max_drawdown
                }
            }
            earning = round( round(predicted_avg_return * 12, 5) / round(daily_avg_return * 12, 5),2)
            # 設置 context，將要顯示的資料放入其中
            context = {
                'show': final_result,  # 最終預測結果，如 weighted average 與 final prediction=> need to change to "final_result"！！
                'show3': prediction_text3,
                'show5': prediction_text5,
                'chart_path': 'static/images/monthly_return_comparison.png',  # 圖像路徑
                'comparison': comparison,  # 策略比較結果     
                'earning': earning   
            }
            print(chart_path)
            return render(request, 'page5_rolling_split.html', context)
        

    # 若非 POST 請求，顯示頁面
    selected_data = request.session.get('train_data') if request.session.get('train_data') else request.session.get('test_data')
    columns = selected_data[0].keys() if selected_data else []
    return render(request, 'page3_train_test_comparison.html', {'data': selected_data, 'columns': columns})


def page4_bert_model(request):
    if request.method == "POST":
        # 从 session 中获取 JSON 数据
        train_df_json = request.session.get('train_data')
        test_df_json = request.session.get('test_data')

        # 將資料轉回 DataFrame（如果需要）
        train_df = pd.DataFrame(train_df_json)
        test_df = pd.DataFrame(test_df_json)
        target_col = request.session.get('target_slt')

        # 平衡train数据集
        train_majority = train_df[train_df[target_col] == 0]
        train_minority = train_df[train_df[target_col] == 1]

        train_minority_upsampled = resample(train_minority,
                                            replace=True,
                                            n_samples=len(train_majority),
                                            random_state=42)

        train_balanced = pd.concat([train_majority, train_minority_upsampled])

        # 从train数据集中进一步分割出验证集
        X_train = train_balanced['title']
        y_train = train_balanced[target_col]

        X_test = test_df['title']
        y_test = test_df[target_col]

        # 再次从train数据集中进行分层抽样切割，划分为训练集和验证集
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify=y_train, test_size=0.1, random_state=42)
        # 設定模型訓練參數
        your_num_labels = 2
        your_max_length = 50
        your_batch_size = 64 #128好像比較好。
        your_epochs = 10

        # 使用 CKIP 提供的 BERT 模型
        model_name = 'ckiplab/bert-base-chinese'

        # 初始化 tokenizer 和 model
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        bert_model = AutoModel.from_pretrained(model_name)

        # 定义包含 BERT 和分类头的模型
        class BertForClassification(Module):
            def __init__(self, bert_model, num_labels):
                super(BertForClassification, self).__init__()
                self.bert = bert_model
                self.classifier = Linear(self.bert.config.hidden_size, num_labels)

            def forward(self, input_ids, attention_mask):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs[1]  # 使用池化后的输出
                logits = self.classifier(pooled_output)
                return logits
        # 3. 增加數據增強
        def augment_data(texts):
            augmented_texts = []
            for text in texts:
                # 這裡可以加入數據增強方法，如隨機刪詞、同義詞替換等
                augmented_texts.append(text)
            return augmented_texts

        X_train = augment_data(X_train)
        X_valid = augment_data(X_valid)
        X_test = augment_data(X_test)
        print(torch.cuda.is_available())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_labels = 2  # 假设是二分类任务
        model = BertForClassification(bert_model, num_labels)  
        print("Using device:", device)
        model = model.to(device)   
        # 將數據轉換為 BERT 所需的格式
        def encode_data(texts, labels, max_length):
            input_ids = []
            attention_masks = []
            for text in texts:
                encoded = tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                input_ids.append(encoded['input_ids'])
                attention_masks.append(encoded['attention_mask'])

            input_ids = torch.cat(input_ids, dim=0)
            attention_masks = torch.cat(attention_masks, dim=0)
            labels = torch.tensor(labels.values)

            return TensorDataset(input_ids, attention_masks, labels)

        train_dataset = encode_data(X_train, y_train, your_max_length)
        valid_dataset = encode_data(X_valid, y_valid, your_max_length)
        test_dataset = encode_data(X_test, y_test, your_max_length)

        train_dataloader = DataLoader(train_dataset, batch_size=your_batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=your_batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=your_batch_size)

        # 防止模型過擬合 如果驗證集開始停滯就停止
        class EarlyStopping:
            def __init__(self, patience=2, delta=0):
                self.patience = patience
                self.delta = delta
                self.best_loss = np.inf
                self.wait = 0
                self.early_stop = False

            def __call__(self, val_loss):
                if self.best_loss - val_loss > self.delta:
                    self.best_loss = val_loss
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.early_stop = True

        def train_model(model, train_dataloader, valid_dataloader, epochs=10):#3 9.6=>0.73-0.5698 7.6=>0.6677-0.6536 =>batch=8
            early_stopping = EarlyStopping(patience=2, delta=0.01)    #3 9.6=>0.7-0.6453 1.5=>0.6739-0.648 8.6=>0.6398-0.6508 7.6=>0.6708-0.6508 =>batch16
            optimizer = torch.optim.AdamW(model.parameters(), lr=9e-6) #1 9.6=>0.75-0.592 1.5=>0.735-0.573 8.6=>0.74-0.56
            criterion = torch.nn.CrossEntropyLoss() #5 9.6=>0.6801-0.6844 8.6=>0.6708-0.6732 7.6=>0.6708-0.6704 1.5=>0.6646-0.6453
            model.train()
            
            for epoch in range(epochs):
                print(f'Epoch {epoch + 1}/{epochs}')
                total_loss = 0

                for batch in tqdm(train_dataloader, desc='Training'):
                    batch_input_ids, batch_attention_masks, batch_labels = [t.to(device) for t in batch]
                    batch_labels = batch_labels.long()  # 转换为 LongTensor
                    model.zero_grad()
                    outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks)
                    logits = outputs  # 获取输出
                    loss = criterion(logits, batch_labels)
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                avg_train_loss = total_loss / len(train_dataloader)
                print(f'Average training loss: {avg_train_loss}')

                # 评估
                model.eval()
                eval_loss = 0
                eval_accuracy = 0
                nb_eval_steps = 0

                for batch in valid_dataloader:
                    batch_input_ids, batch_attention_masks, batch_labels = [t.to(device) for t in batch]
                    batch_labels = batch_labels.long()  # 转换为 LongTensor
                    with torch.no_grad():
                        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks)
                        logits = outputs
                        loss = criterion(logits, batch_labels)
                        eval_loss += loss.item()

                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                        batch_labels = batch_labels.cpu().numpy()
                        accuracy = (preds == batch_labels).mean() * 100
                        eval_accuracy += accuracy
                        nb_eval_steps += 1

                avg_val_accuracy = eval_accuracy / nb_eval_steps
                avg_val_loss = eval_loss / nb_eval_steps

                print(f'Validation loss: {avg_val_loss}')
                print(f'Validation accuracy: {avg_val_accuracy}')

                # Early stopping
                early_stopping(avg_val_loss)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        # 訓練模型
        train_model(model, train_dataloader, valid_dataloader, epochs=your_epochs)
        
        def evaluate_model(dataloader):
            model.eval()
            loss = 0.0
            predictions = []
            labels = []

            criterion = torch.nn.CrossEntropyLoss()
            with torch.no_grad():
                for batch in tqdm(dataloader, desc='Evaluating'):
                    input_ids, attention_mask, labels_batch = batch
                    input_ids, attention_mask, labels_batch = input_ids.to(device), attention_mask.to(device), labels_batch.to(device)

                    logits = model(input_ids, attention_mask=attention_mask)

                    loss += criterion(logits, labels_batch.long()).item()

                    _, predicted = torch.max(logits, 1)
                    predictions.extend(predicted.cpu().numpy())
                    labels.extend(labels_batch.cpu().numpy())

            avg_loss = loss / len(dataloader)
            accuracy = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions, average='weighted')
            recall = recall_score(labels, predictions, average='weighted')
            f1 = f1_score(labels, predictions, average='weighted')
            auc = roc_auc_score(labels, predictions, average='weighted')

            return avg_loss, accuracy, precision, recall, f1, auc, labels, predictions

        # 混淆矩陣繪製函數
        def plot_confusion_matrix(labels, predictions, title, filename):
            if not isinstance(labels, (list, np.ndarray)) or not isinstance(predictions, (list, np.ndarray)):
                raise ValueError("Labels and predictions must be lists or numpy arrays.")
            if len(labels) != len(predictions):
                raise ValueError("Labels and predictions must be of the same length.")
            
            conf_matrix = confusion_matrix(labels, predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(title)
            
            static_dir = os.path.join(settings.STATICFILES_DIRS[0], 'images')
            os.makedirs(static_dir, exist_ok=True)
            file_path = os.path.join(static_dir, filename)
            
            plt.savefig(file_path)
            plt.close()
            return file_path


        # 假设evaluate_model函数返回以下指标
        train_results = evaluate_model(train_dataloader)
        train_loss, train_accuracy, train_precision, train_recall, train_f1, train_auc, train_labels, train_predictions = train_results
        
        valid_results = evaluate_model(valid_dataloader)
        valid_loss, valid_accuracy, valid_precision, valid_recall, valid_f1, valid_auc, valid_labels, valid_predictions = valid_results
        
        test_results = evaluate_model(test_dataloader)
        test_loss, test_accuracy, test_precision, test_recall, test_f1, test_auc, test_labels, test_predictions = test_results
        
        # 保存並返回混淆矩陣圖像路徑
        train_cm_path = plot_confusion_matrix(train_labels, train_predictions, 'Confusion Matrix - Train', 'Confusion_Matrix_Train.png')
        valid_cm_path = plot_confusion_matrix(valid_labels, valid_predictions, 'Confusion Matrix - Validation', 'Confusion_Matrix_Validation.png')
        test_cm_path = plot_confusion_matrix(test_labels, test_predictions, 'Confusion Matrix - Test', 'Confusion_Matrix_Test.png')

        # 渲染模板并传递数据
        return render(request, 'page4_bert_model.html', {
            'train_accuracy': train_accuracy,
            'valid_accuracy': valid_accuracy,
            'test_accuracy': test_accuracy,
            'train_cm_path': train_cm_path,
            'valid_cm_path': valid_cm_path,
            'test_cm_path': test_cm_path,
        })
    else:
        # 如果请求不是 POST 方法，返回一个错误页面或重定向到其他页面
        return render(request, 'page4_bert_model.html')
    
dash = My_Dash()
    
def home(request):
    return render(request, "page0_home.html")


def page1_1_visualization(request):
    import matplotlib
    matplotlib.use('Agg')  # 使用 Agg 後端
    import matplotlib.pyplot as plt
    # 獲取目前腳本檔案的目錄
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # 獲取上一層目錄
    parent_directory = os.path.dirname(current_directory)

    # 定義 "all.csv" 的路徑
    file_path = os.path.join(current_directory, 'all.csv')

    # 定義 CKIP WS 的路徑，位於上一層資料夾
    ckip_path = os.path.join(parent_directory, 'data')
    # 检查文件是否存在
    if os.path.exists(file_path) and request.method == 'POST' and 'data_upload_button' in request.POST:
        try:
            # 读取 "all.csv"
            dash.df1 = pd.read_csv(file_path)

            if dash.df1 is not None and not dash.df1.empty:
                # 進行你想要的處理

                # 篩選出 next_ans 等於 1 的行
                filtered_df_next_ans_1 = dash.df1[dash.df1['next_ans'] == 1]

                # 取出篩選後的 title 列，並將其轉換為一個字符串
                text = ' '.join(filtered_df_next_ans_1['title'].tolist())

                # 斷詞，設置CKIP WS的路徑
                ws = WS(ckip_path)
                word_sentence_list = ws([text])

                # 將斷詞結果轉換為字符串
                word_sentence_str = ' '.join([' '.join(sentence) for sentence in word_sentence_list])

                # 定義你想要排除的關鍵字
                stopwords = set(['鴻海', '台股盤前', '台股盤', '台股', '盤前', '股盤', '台', '今日', '必看', '財經', '新聞'])

                # 指定字體路徑，替換為系統中的中文字體路徑
                font_path = r'C:\Windows\Fonts\msjh.ttc'

                # 生成文字雲，使用字符串而不是列表
                wordcloud = WordCloud(font_path=font_path, width=800, height=400, background_color='white', stopwords=stopwords).generate(word_sentence_str)

                # 視覺化文字雲
                plt.figure(figsize=(12, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')  # 不顯示坐標
                # 在生成完文字雲後，保存為圖片並設置圖片路徑
                images_dir = os.path.join('static', 'images')
                wordcloud_image_path = os.path.join(images_dir, 'wordcloud_image.png')
                plt.savefig(wordcloud_image_path, format='png')  # 保存圖片
                plt.close() 
                # 確保 'date' 被解析為日期格式
                dash.df1['date'] = pd.to_datetime(dash.df1['date'])
                # 按照 'date' 列進行排序
                dash.df1 = dash.df1.sort_values(by='date')

                # 如果你希望重設索引，則可以使用 reset_index() 方法
                dash.df1 = dash.df1.reset_index(drop=True)
                # 繪製圖表
                plt.figure(figsize=(12, 6))

                # 繪製 'Open' 的折線圖
                plt.plot(dash.df1['date'], dash.df1['Open'], label='Open Price', color='blue', linestyle='-', marker='o')
                # 添加圖表標題和標籤
                plt.title('Stock Open Prices Over Time')
                plt.xlabel('Date')
                plt.ylabel('Open Price')
                # 顯示圖例
                plt.legend()
                # 美化日期顯示
                plt.xticks(rotation=45)
                # 顯示圖表
                plt.grid(True)
                plt.tight_layout()
                images_dir = os.path.join('static', 'images')
                openprice_image_path = os.path.join(images_dir, 'Open_price.png')
                plt.savefig(openprice_image_path, format='png')  # 保存圖片
                plt.close() 

                return render(request, 'page1_1_visualization.html', {'wordcloud_image_url': wordcloud_image_path, 'openprice_image_url': openprice_image_path, 'display': True})
            else:
                messages.warning(request, 'The file "all.csv" is empty or failed to load.')
                return render(request, 'page1_crawler.html')  # 返回到文件上传页面
        except Exception as e:
            # 捕获异常，打印异常信息并返回到错误页面
            print(f"Error during file processing: {e}")
            messages.error(request, 'An error occurred while processing the file.')
            return render(request, 'page1_crawler.html')