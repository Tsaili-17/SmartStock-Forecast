from django.db import models
import plotly.graph_objects as go
import pandas as pd
import io
from sklearn.model_selection import train_test_split

class My_Dash:
    def __init__(self):
        self.df1 = None
        self.df2 = None
        
        self.train_df = None
        self.valid_df = None
        self.test_df = None
        
        self.categorical_col = None
        self.numerical_col = None
        self.target_col = None
        
        self.test_numerical_col = None
        self.test_categorical_col = None
        
        self.target_is_numerical = None
        self.target_is_categorical = None
        self.class_mapping = None
        
        self.target_in_test = None
        
        self.slt_feat = None
        
        self.scaling_method = False
        self.feature_selection = False
        self.one_hot_feat = None
        self.pca_flag = False
        
        self.optimal_params = None
        self.use_optimal_params = False
        
        self.imbalanced = False
        self.imbalanced_class_mapping = None
        self.imbalanced_class_percentage = None
        self.imbalanced_encoded_class_percentage = None
        
    def any_none(self, *args):
        """检查传入的参数中是否有 None"""
        return any(arg is None for arg in args)
    
    def placeholder_figure(self):
        """创建一个占位图形"""
        placeholder_figure = go.Figure(data=[],
                                       layout=go.Layout(xaxis=dict(title='X Axis Title'),
                                                        yaxis=dict(title='Y Axis Title'),
                                                        margin=dict(l=75, r=75, t=10, b=20),
                                                        width=1000,
                                                        height=500)
        )
        return placeholder_figure

class data_decoded:
    def __init__(self, data):
        self.data = data
    
    def file_decode(self):
        """解码上传的文件并返回数据框"""
        file_extension = self.data.name.split('.')[-1].lower()
        data = self.data.read()
        
        if file_extension == 'csv':
            data = pd.read_csv(io.StringIO(data.decode('utf-8')))
        elif file_extension == 'xlsx':
            data = pd.read_excel(io.BytesIO(data))
        elif file_extension == 'json':
            data = pd.read_json(data.decode('utf-8'))
        else:
            raise ValueError(f'File Type Unsupported: {file_extension}')
        
        return data
