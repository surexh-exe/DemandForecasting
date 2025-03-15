Demand Forecasting for Grocery Stores with Sales Advisor Chatbot

📌 Introduction

This project provides a demand forecasting system for grocery stores, leveraging various machine learning and deep learning models such as XGBoost, ARIMA, and Prophet. Additionally, it includes an AI-powered Sales Advisor Chatbot that provides insights and recommendations based on sales data trends.

⚡ Features

Sales Data Preprocessing: Cleans and processes store sales data.

Forecasting Models: Uses ARIMA, Prophet, and XGBoost for time-series forecasting.

Data Visualization: Generates sales trend plots for better insights.

AI-Powered Chatbot: Uses Gemini AI to provide recommendations based on trends.

🛠 Installation & Setup

1️⃣ Prerequisites

Ensure you have the following installed:

Python 3.8+

Pip

Virtual environment (optional but recommended)

2️⃣ Clone the Repository

git clone https://github.com/yourusername/demand-forecasting-chatbot.git
cd demand-forecasting-chatbot

3️⃣ Create a Virtual Environment (Optional)

python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows

4️⃣ Install Dependencies

pip install -r requirements.txt

🚀 Running the Application

1️⃣ Prepare the Dataset

Ensure that the following CSV files are available inside the data/ folder:

Rossmann_Stores_Data.csv (Sales data)

store.csv (Store details)

2️⃣ Run the Forecasting Script

python main.py

The script will:

Load and preprocess sales and store data.

Train ARIMA, Prophet, and XGBoost models.

Generate sales trend visualizations.

Use AI (Gemini API) to provide insights.

3️⃣ View the AI Sales Advisor Insights

After execution, the chatbot will generate strategic insights about sales trends, inventory management, and forecasting accuracy. You will see output like:

🤖 Sales Advisor Insights:
- Expected demand increase in Q2 due to seasonal trends.
- Inventory adjustment recommendations to reduce stockout risks.
- Suggested promotional offers based on past sales trends.

4️⃣ Visualize the Sales Trend

The generated sales trend plot is saved in Images/monthly_sales_trend.png. Open it to analyze the trends.

🏆 Key Components Explained

🔹 Machine Learning Models

ARIMA: Used for time-series forecasting.

Prophet: Facebook’s open-source model for time-series analysis.

XGBoost: A powerful gradient boosting model optimized for regression.

🔹 AI-Powered Sales Advisor Chatbot

Uses Google’s Gemini AI to generate sales insights.

Accepts sales data and trend visualizations as input.

Provides recommendations on inventory and demand trends.

❓ Troubleshooting & FAQs

1️⃣ "ModuleNotFoundError: No module named 'tensorflow'"

Run:

pip install tensorflow

2️⃣ "FileNotFoundError: No such file or directory: 'data/Rossmann_Stores_Data.csv'"

Ensure that your dataset files are placed in the data/ folder.

3️⃣ How to update API key for Gemini AI?

Replace the API key in main.py:

API_KEY = "your_new_api_key_here"
genai.configure(api_key=API_KEY)

👨‍💻 Contributing

Feel free to contribute to the project! Fork the repo, make changes, and submit a pull request.

🏆 Acknowledgments

Data Source: Rossmann Stores Dataset

Libraries: Pandas, NumPy, TensorFlow, Prophet, XGBoost

AI API: Google Gemini

📜 License

This project is licensed under the MIT License.

📬 Contact

For any questions or issues, feel free to reach out via GitHub Issues.
