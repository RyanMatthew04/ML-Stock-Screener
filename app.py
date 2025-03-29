import streamlit as st
import pandas as pd
from svm import svm
from rf import rf
from finbert import finbert
import plotly.graph_objects as go
import numpy as np
st.markdown(
    """
    <style>
    .header-center {
        text-align: center;
    }
    div.stButton > button {
    color: white !important; /* Ensures text stays white */
    background-color: #C70039; /* Default background */
    transition: all 0.3s ease-in-out;
    }

    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        background-color: #A0002D !important; /* Darker shade on hover */
        color: white !important; /* Ensures text remains visible */
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

df_merged = pd.read_csv('artifacts/merged.csv')
df_merged["Signal"] = np.where(df_merged["bullish_probability"] > df_merged["bearish_probability"], "Buy", "Sell")
df_merged["Confidence"] = (df_merged[["bullish_probability", "bearish_probability"]].max(axis=1) * 100).round(2).astype(str) + "%"

st.sidebar.markdown("<h2 style='font-size:24px;'>Filter Strategy</h2>", unsafe_allow_html=True)
signal_options = ["Buy", "Sell"]
selected_dir = st.sidebar.selectbox("Signal", signal_options)
df_merged = df_merged[df_merged['Signal']==selected_dir]
decimal_value = st.sidebar.slider(
    "Confidence Threshold",
    min_value=50.0, 
    max_value=100.0, 
    value=50.0, 
    step=1.0, 
    format="%.2f"
)

df_merged = df_merged[(df_merged[["bullish_probability", "bearish_probability"]].max(axis=1) * 100).round(2) >= decimal_value]
df_merged['max_prob'] = df_merged[['bullish_probability', 'bearish_probability']].max(axis=1)
df_merged = df_merged.sort_values(by='max_prob', ascending=False)
df_merged = df_merged[['Stock', 'Signal', 'Confidence']]

if st.sidebar.button("Fetch", use_container_width=True):
    df_svm = svm()      
    df_rf = rf()
    df_bert = finbert()

    # Use outer join to retain all 50 rows
    merged_df = pd.merge(df_svm, df_rf, on='Stock', how='outer')
    merged_df = pd.merge(merged_df, df_bert, on='Stock', how='outer')

    # Calculate the weighted probabilities
    merged_df['bearish_probability'] = (
        merged_df['RF_Bearish_Probability'] +
         merged_df['SVM_Bearish_Probability'] +
        merged_df['BERT_Bearish_Probability']
    ) / 3

    merged_df['bullish_probability'] = (
        merged_df['RF_Bullish_Probability'] +
         merged_df['SVM_Bullish_Probability'] +
        merged_df['BERT_Bullish_Probability']
    )/3

    # Save the merged DataFrame
    merged_df.to_csv('artifacts/merged.csv', index=False)



st.markdown(f'<h1 class="header-center">NIFTY 50 Screener</h1>', unsafe_allow_html=True)
fig = go.Figure(data=[go.Table(
    columnorder = [1,2,3],
    columnwidth = [100,100,100],
    header=dict(
        values=list(df_merged.columns),
        fill_color="black",  
        font=dict(color="white", size=13),
        align="left"
    ),
    cells=dict(
        values=[df_merged[col] for col in df_merged.columns],
        fill_color="#1E1E1E", 
        font=dict(color="white", size=12),
        align="left",
        height=30  
    )
)])
fig.update_layout(
height = 440,
margin=dict(t=40,b=40)  
)

st.plotly_chart(fig)


