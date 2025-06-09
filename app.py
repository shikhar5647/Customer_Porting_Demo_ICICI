import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CDC: Customer Porting Insight Report",
    page_icon="💰",
    layout="wide"
)

# --- GEMINI API CONFIGURATION (No changes here) ---
def configure_api():
    """Configure the Gemini API with the key from Streamlit secrets."""
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
        return True
    except KeyError:
        st.error("🔑 Google API Key not found. Please add it to your Streamlit secrets!")
        st.info("Create a .streamlit/secrets.toml file with: GOOGLE_API_KEY = 'Your_API_Key'")
        return False
    except Exception as e:
        st.error(f"An error occurred during API configuration: {e}")
        return False

# --- DATA LOADING AND PREPROCESSING (No changes here) ---
@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """Loads and preprocesses the transaction data from a CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
        # Data Cleaning and Transformation
        df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
        df['Debit_Amount'] = pd.to_numeric(df['Debit_Amount'], errors='coerce').fillna(0)
        df['Credit_Amount'] = pd.to_numeric(df['Credit_Amount'], errors='coerce').fillna(0)
        df['Month'] = df['Transaction_Date'].dt.to_period('M').astype(str)
        # Correcting potential data entry errors in the sample
        df.loc[df['Description'].str.contains('Freelance|Dividend|Refund|Interest', case=False, na=False), 'Transaction_Type'] = 'Credit'
        return df
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return None

# --- AI-POWERED ANALYSIS ---
def get_gemini_insights(df):
    """Generates financial insights using the Gemini API."""
    if len(df) > 500:
        data_sample = df.sample(500).to_csv(index=False)
    else:
        data_sample = df.to_csv(index=False)

    model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
    prompt = f"""
    You are an expert financial analyst. Your task is to provide a detailed financial insight report based on the following customer transaction data.
    The data is provided in CSV format below.

    **Data:**
    {data_sample}

    **Instructions:**
    Analyze the data and generate a comprehensive report in Markdown format. The report should include the following sections:

    1.  **Overall Financial Summary:** A brief overview of the customer's financial health, including key trends in income, spending, and savings.
    2.  **Income Analysis:**
        *   Identify all primary and secondary sources of income (e.g., Salary, Freelance, Investments).
        *   Comment on the stability and frequency of income.
    3.  **Spending Habits Analysis:**
        *   Break down the major spending categories (e.g., Housing, EMI, Food, Utilities, Shopping).
        *   Identify the top 3-5 spending categories by amount.
        *   Point out any significant or discretionary spending patterns.
    4.  **Loan and Debt Analysis:**
        *   Identify all loan payments (EMIs). Specify the type of loan if possible (e.g., Home Loan, Car Loan).
        *   Comment on the proportion of income going towards debt repayment.
    5.  **Investment & Savings:**
        *   Identify any investments like Mutual Fund SIPs or Fixed Deposits.
        *   Provide an assessment of the customer's saving and investment discipline.
    6.  **Actionable Recommendations:**
        *   Provide 3-5 clear, actionable recommendations for improving financial health. This could include budgeting tips, saving strategies, or debt management advice.

    Structure your response clearly with headers for each section. Be insightful and professional.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Could not generate insights due to an error: {e}"

# --- MAIN APPLICATION LOGIC ---
def main():
    st.title("📊 CDC: Customer Porting Insight Report")
    st.markdown("Upload your bank transaction CSV file to generate a personalized financial health report.")

    if not configure_api():
        st.stop()
        
    # --- VISUALIZATION STYLING (New section) ---
    # Set a theme for seaborn that works well with dark backgrounds
    sns.set_theme(style="darkgrid")
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'


    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Please upload a CSV file with columns like 'Transaction_Date', 'Description', 'Debit_Amount', 'Credit_Amount', 'Category'"
    )

    if uploaded_file is not None:
        df = load_and_preprocess_data(uploaded_file)

        if df is not None:
            st.success(f"Successfully loaded {len(df)} transactions.")
            
            tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Financial Advisor", "📈 Income vs. Spending", "📊 Category Deep-Dive", "🔄 Transaction Flow"])

            with tab1:
                st.header("🤖 AI-Powered Financial Advisor")
                with st.spinner("Your personal AI advisor is analyzing your finances..."):
                    insights = get_gemini_insights(df)
                    st.markdown(insights)

            with tab2:
                st.header("📈 Monthly Income vs. Spending")
                monthly_summary = df.groupby('Month').agg(
                    Total_Income=('Credit_Amount', 'sum'),
                    Total_Spending=('Debit_Amount', 'sum')
                ).reset_index()

                # --- CHART MODIFICATION: Monthly Summary ---
                # We need to "melt" the dataframe to make it suitable for seaborn's grouped bar plot
                monthly_summary_melted = monthly_summary.melt(id_vars='Month', var_name='Transaction Type', value_name='Amount')
                
                fig_monthly, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    data=monthly_summary_melted,
                    x='Month',
                    y='Amount',
                    hue='Transaction Type',
                    palette={'Total_Income': '#00B894', 'Total_Spending': '#D63031'},
                    ax=ax
                )
                ax.set_title('Monthly Income vs. Spending Overview')
                ax.set_ylabel('Amount')
                ax.tick_params(axis='x', rotation=45)
                fig_monthly.patch.set_alpha(0) # Make background transparent for Streamlit theme
                ax.patch.set_alpha(0)
                st.pyplot(fig_monthly)
                st.dataframe(monthly_summary, use_container_width=True)

            with tab3:
                st.header("📊 Spending and Income Category Analysis")
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Spending by Category")
                    spending_by_cat = df[df['Debit_Amount'] > 0].groupby('Category')['Debit_Amount'].sum().sort_values(ascending=False)
                    
                    # --- CHART MODIFICATION: Spending Pie ---
                    fig_spending, ax = plt.subplots(figsize=(8, 8))
                    ax.pie(
                        spending_by_cat,
                        labels=spending_by_cat.index,
                        autopct='%1.1f%%',
                        startangle=90,
                        wedgeprops=dict(width=0.4), # This creates the donut hole
                        pctdistance=0.8
                    )
                    ax.set_title('Spending Distribution')
                    fig_spending.patch.set_alpha(0)
                    ax.patch.set_alpha(0)
                    st.pyplot(fig_spending)
                    st.dataframe(spending_by_cat.reset_index(), use_container_width=True)

                with col2:
                    st.subheader("Income by Category")
                    income_by_cat = df[df['Credit_Amount'] > 0].groupby('Category')['Credit_Amount'].sum().sort_values(ascending=False)
                    
                    # --- CHART MODIFICATION: Income Pie ---
                    fig_income, ax = plt.subplots(figsize=(8, 8))
                    ax.pie(
                        income_by_cat,
                        labels=income_by_cat.index,
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=sns.color_palette('Greens_r', len(income_by_cat)),
                        wedgeprops=dict(width=0.4),
                        pctdistance=0.8
                    )
                    ax.set_title('Income Sources')
                    fig_income.patch.set_alpha(0)
                    ax.patch.set_alpha(0)
                    st.pyplot(fig_income)
                    st.dataframe(income_by_cat.reset_index(), use_container_width=True)

            with tab4:
                # --- NO CHANGES in this tab, as it only displays dataframes ---
                st.header("🔄 Frequent Transaction Flow")
                col3, col4 = st.columns(2)

                with col3:
                    st.subheader("Top 10 Outflows (Debits)")
                    outflows = df[df['Debit_Amount'] > 0].groupby('Beneficiary_Name')['Debit_Amount'].agg(['sum', 'count']).sort_values(by='sum', ascending=False).head(10)
                    outflows = outflows.rename(columns={'sum': 'Total Amount', 'count': 'Frequency'})
                    st.dataframe(outflows, use_container_width=True)

                with col4:
                    st.subheader("Top 10 Inflows (Credits)")
                    df['Inflow_Source'] = df.apply(lambda row: row['Beneficiary_Name'] if pd.notna(row['Beneficiary_Name']) else row['Description'], axis=1)
                    inflows = df[df['Credit_Amount'] > 0].groupby('Inflow_Source')['Credit_Amount'].agg(['sum', 'count']).sort_values(by='sum', ascending=False).head(10)
                    inflows = inflows.rename(columns={'sum': 'Total Amount', 'count': 'Frequency'})
                    st.dataframe(inflows, use_container_width=True)


if __name__ == "__main__":
    main()