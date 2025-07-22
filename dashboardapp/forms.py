from django import forms

class StockCrawlForm(forms.Form):
    keyword = forms.CharField(label='Stock Name', max_length=100)
    #start_date = forms.DateField(label='Start Date', widget=forms.SelectDateWidget)
    #end_date = forms.DateField(label='End Date', widget=forms.SelectDateWidget)
    stock_codes = forms.CharField(label='Stock Code', max_length=100)
