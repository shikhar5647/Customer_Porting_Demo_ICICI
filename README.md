# Customer Porting Financial Insight Report (ICICI Demo)

[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)](https://streamlit.io)

An intelligent financial analysis tool that transforms raw bank transaction data into a comprehensive and actionable insight report. This application leverages the power of Google's Gemini API for advanced financial analysis and presents the findings through an intuitive Streamlit web interface.

This project serves as a demonstration of how a customer's financial data can be analyzed to provide valuable insights, potentially for use cases like customer porting or personalized financial advisory services.

## 🚀 Overview

The goal of this project is to provide users with a deep understanding of their financial health by simply uploading a CSV of their bank transactions. The application automatically processes the data, generates insightful visualizations, and provides an AI-powered narrative report with personalized recommendations.

## ✨ Key Features

-   **🤖 AI Financial Advisor:** Utilizes the Google Gemini Pro API to generate a detailed, human-like report covering income sources, spending habits, loan analysis, and investment patterns.
-   **📈 Interactive Dashboard:** A clean, tab-based interface to explore different facets of your financial data.
-   **📊 Rich Visualizations:** Dynamic charts for monthly income vs. spending, and clear donut charts for spending/income category breakdowns, built with Matplotlib and Seaborn.
-   **🔄 Transaction Flow Analysis:** Identifies and ranks the most frequent and highest-value incoming and outgoing transactions.
-   **🔒 Secure & Private:** Your data is processed for each session and is not stored. API keys are managed securely using Streamlit's secrets management.
-   **🎨 Custom Theming:** A sleek, custom dark-orange theme for a professional look and feel.


---

## 🛠️ Technology Stack

-   **Frontend:** [Streamlit](https://streamlit.io/)
-   **Backend & Data Logic:** [Python](https://www.python.org/), [Pandas](https://pandas.pydata.org/)
-   **AI Engine:** [Google Gemini Pro API](https://ai.google.dev/)
-   **Data Visualization:** [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/)

---

## ⚙️ Setup and Installation

Follow these steps to get the application running on your local machine.

### Prerequisites

-   Python 3.8 or higher
-   A Google Gemini API Key. You can get one from the [Google AI Studio](https://aistudio.google.com/app/apikey).

### 1. Clone the Repository

```bash
git clone https://github.com/shikhar5647/Customer_Porting_Demo_ICICI.git
cd Customer_Porting_Demo_ICICI
```
