# Demand Forecasting for Grocery Stores with Sales Advisor Chatbot

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import xgboost as xgb
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import google.generativeai as genai
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up Gemini AI
API_KEY = "AIzaSyDrDQeg51SeCJf_iepr8aRT0udL3piLNkU"
genai.configure(api_key=API_KEY)

# Set file paths for local environment
base_path = os.path.dirname(__file__)

# Load datasets
sales_data_path = os.path.join(base_path, 'data', 'Rossmann_Stores_Data.csv')
store_data_path = os.path.join(base_path, 'data', 'store.csv')
print(f"Loading sales data from {sales_data_path}")
print(f"Loading store data from {store_data_path}")
sales_df = pd.read_csv(sales_data_path, low_memory=False)
store_df = pd.read_csv(store_data_path, low_memory=False)

# Merge datasets
df = pd.merge(sales_df, store_df, on='Store', how='left')

# Preprocess data
def preprocess_data(df):
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    
    # Detect date and sales columns dynamically
    def detect_columns(df):
        date_col = None
        sales_col = None
        for col in df.columns:
            if "date" in col.lower():
                date_col = col
            if "sales" in col.lower():
                sales_col = col
        if not date_col or not sales_col:
            raise ValueError(f"Unable to detect date or sales columns. Available columns: {list(df.columns)}")
        return date_col, sales_col
    
    date_col, sales_col = detect_columns(df)
    
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')
    df = df.dropna(subset=[date_col, sales_col])  # Ensure valid date and sales values
    
    df['month'] = df[date_col].dt.to_period('M').astype(str)
    df = df.groupby('month', as_index=False)[sales_col].sum()
    
    return df, 'month', sales_col

df, date_col, sales_col = preprocess_data(df)

# Dataset First Look
print("First Look at Sales Data:")
print(sales_df.head())
print(sales_df.info())
print(sales_df.tail())
print(sales_df.describe())

print("\nState Holiday Value Counts:")
print(sales_df['StateHoliday'].value_counts())

print("\nUnique Dates:")
print(sales_df['Date'].unique())

print("\nMissing Values in Sales Data:")
print(sales_df.isnull().sum())

print("\nStore Data Info:")
print(store_df.info())

print("\nUnique School Holidays:")
print(sales_df['SchoolHoliday'].unique())

print("\nMissing Values in Store Data:")
print(store_df.isnull().sum())

# Enhanced Data Visualization
def plot_sales_trend(df, date_col, sales_col):
    try:
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=df[date_col], y=df[sales_col], marker='o', linestyle='-')
        plt.title('Monthly Sales Trend')
        plt.xlabel('Month')
        plt.ylabel('Sales')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.savefig('Images/monthly_sales_trend.png')
        plt.show()
    except Exception as e:
        print(f"Error in plot_sales_trend: {e}")

plot_sales_trend(df, date_col, sales_col)

# Forecasting Models
def train_arima(df, sales_col):
    try:
        model = ARIMA(df[sales_col], order=(2,1,2))
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        logging.error(f"Error in train_arima: {e}")

def train_prophet(df, date_col, sales_col):
    try:
        df_prophet = df.rename(columns={date_col: 'ds', sales_col: 'y'})
        model = Prophet()
        model.fit(df_prophet)
        return model
    except Exception as e:
        logging.error(f"Error in train_prophet: {e}")

def train_xgboost(df, sales_col):
    try:
        df['month_num'] = pd.to_datetime(df['month']).dt.month
        df['year'] = pd.to_datetime(df['month']).dt.year
        X = df[['month_num', 'year']]
        y = df[sales_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBRegressor(objective='reg:squarederror')
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        logging.error(f"Error in train_xgboost: {e}")

# AI-Driven Sales Advisor Chatbot
def sales_advisor_chatbot(sales_data, images):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"You are a sales forecasting advisor. Based on the following sales data and visualizations, provide strategic insights, potential demand trends, and suggestions for optimizing inventory and sales strategies:\n\n{sales_data.to_string()}\n\nVisualizations:\n"
        for image in images:
            prompt += f"![{image}]({image})\n"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error in sales_advisor_chatbot: {e}")

# Train Models and Get AI Insights
arima_model = train_arima(df, sales_col)
prophet_model = train_prophet(df, date_col, sales_col)
xgboost_model = train_xgboost(df, sales_col)

# Get AI-generated sales advisory insights
images = [
    'Images/monthly_sales_trend.png', 
]
insights = sales_advisor_chatbot(df, images)
print("\n🤖 Sales Advisor Insights:\n")
print(insights)
