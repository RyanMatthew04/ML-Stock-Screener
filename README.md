# Multimodal Stock Reversal Prediction System

This repository contains a multimodal stock reversal prediction system for Nifty 50 stocks. The system ensembles three distinct predictive models—a Random Forest classifier, a Support Vector Classifier, and FinBERT's bullish/bearish sentiment probabilities—using soft voting via simple averaging. It dynamically fetches and processes the latest data, providing actionable trading signals along with confidence scores.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset Sources and Description](#dataset-sources-and-description)
- [Solution Details](#solution-details)
  - [Technical Indicators & Feature Engineering](#technical-indicators--feature-engineering)
  - [Chart Pattern Analysis](#chart-pattern-analysis)
  - [Sentiment Analysis via FinBERT](#sentiment-analysis-via-finbert)
- [System Architecture](#system-architecture)
- [Usage and Execution](#usage-and-execution)
- [Future Enhancements](#future-enhancements)
- [Acknowledgements](#acknowledgements)

---

## Overview

This system is designed to predict reversal points in Nifty 50 stocks by leveraging a multimodal approach that combines technical analysis, chart pattern visualization, and sentiment analysis. By ensembling predictions from these diverse data sources, the system aims to deliver reliable bullish (buy) and bearish (sell) signals that help traders capitalize on key market reversals.

---

## Problem Statement

The stock market is inherently dynamic and complex, posing significant challenges for traders who seek to accurately identify reversal points—critical moments when the prevailing trend is likely to change. Accurately detecting these reversal points is essential for effective risk management and for capitalizing on emerging trends. However, market noise, volatility, and the interplay of numerous influencing factors often obscure these key signals.

This project addresses these challenges by developing a multimodal stock reversal prediction system for Nifty 50 stocks that:

- **Identifies Reversal Dates:** Detects reversal points where the stock's low is lower than the preceding and succeeding 10 candles, and the high is higher than those of the surrounding candles.
- **Extracts Technical Signals:** Computes 40 technical indicators from historical OHLC data (sourced via the Upstox API) using the TA library to train a Random Forest classifier.
- **Analyzes Chart Patterns:** Utilizes the mplfinance library to plot candlestick patterns for identified reversal dates, saving the resulting images which are then used to train a Support Vector Classifier on three-candle peaks and troughs.
- **Incorporates Sentiment Analysis:** Retrieves the latest news headlines for each stock using the GNews library and employs FinBERT as a zero-shot classifier to derive bullish/bearish sentiment probabilities.
- **Ensembles Predictions:** Combines the outputs from the Random Forest classifier, Support Vector Classifier, and FinBERT sentiment analysis through a soft voting mechanism (simple averaging) to deliver a final, dynamic trading signal complete with confidence scores.

By integrating multiple predictive approaches and updating continuously with the latest market data and news, the system offers traders a robust tool to identify potential market reversals with enhanced confidence.

---

## Dataset Sources and Description

### Data Sources

1. **Upstox API (OHLC Data)**
   - **Description:** Provides historical and current Open, High, Low, and Close (OHLC) values for Nifty 50 stocks.
   - **Usage:** Forms the foundational data for computing technical indicators and extracting chart patterns.

2. **TA Library**
   - **Description:** A Python library for technical analysis.
   - **Usage:** Computes 40 technical indicators on reversal dates to serve as features for the Random Forest classifier.

3. **mplfinance Library**
   - **Description:** A library to plot financial charts, specifically candlestick charts.
   - **Usage:** Generates and saves candlestick pattern images of reversal dates that are used to train the Support Vector Classifier.

4. **GNews Library**
   - **Description:** A Python client for fetching the latest news headlines.
   - **Usage:** Retrieves current news headlines for each stock, which are then analyzed for sentiment.

5. **FinBERT**
   - **Description:** A fine-tuned BERT model designed for financial sentiment analysis.
   - **Usage:** Processes news headlines to generate bullish and bearish probabilities, contributing sentiment insights to the ensemble model.

### Dataset Characteristics

- **Reversal Dates:** 
  - Defined as dates where the stock’s low is lower than both the previous 10 and next 10 candles, and the high is higher than those of the surrounding candles.
- **Features:**
  - **Technical Indicators:** 40 indicators computed from OHLC data.
  - **Chart Patterns:** Three-candle peak and trough patterns visualized using mplfinance.
  - **Sentiment Scores:** Bullish and bearish probabilities from FinBERT applied to current news headlines.

---

## Solution Details

### Technical Indicators & Feature Engineering

- **Data Collection:** OHLC data for each stock is fetched using the Upstox API.
- **Indicator Computation:** The TA library is used to calculate 40 technical indicators for each reversal date.
- **Model Training:** A Random Forest classifier is trained on these indicators to predict potential reversal points.

### Chart Pattern Analysis

- **Visualization:** The mplfinance library is used to generate candlestick charts for reversal dates.
- **Image Generation:** Candlestick pattern images are saved and used to extract features based on three-candle peaks and troughs.
- **Model Training:** A Support Vector Classifier (SVC) is trained on these chart patterns to provide an alternative technical perspective.

### Sentiment Analysis via FinBERT

- **News Collection:** Latest news headlines for each stock are collected using the GNews library.
- **Sentiment Classification:** FinBERT processes the headlines as a zero-shot classifier, outputting bullish and bearish probabilities.
- **Integration:** These sentiment scores are integrated with technical predictions to enhance overall forecasting accuracy.

### Ensemble Prediction

- **Soft Voting:** The final prediction is generated by averaging the probabilities output by the Random Forest, Support Vector Classifier, and FinBERT.
- **Dynamic System:** The system dynamically updates with the latest market data and news, providing real-time signals and confidence scores for each stock.

---

## System Architecture

1. **Data Collection Module:**
   - Retrieves historical OHLC data via the Upstox API.
   - Fetches the latest news headlines using the GNews library.

2. **Feature Engineering:**
   - Computes technical indicators using the TA library.
   - Generates candlestick charts and extracts chart patterns using mplfinance.

3. **Model Training and Inference:**
   - **Random Forest Classifier:** Trained on technical indicators.
   - **Support Vector Classifier:** Trained on candlestick chart patterns.
   - **FinBERT:** Provides sentiment analysis from news headlines.
   - **Ensemble Mechanism:** Combines outputs via soft voting to produce final predictions.

4. **User Interface:**
   - Displays the list of Nifty 50 stocks with their respective bullish (buy) or bearish (sell) signals and ensemble confidence scores.
   - Updates dynamically with the latest data.

---

## Usage and Execution

1. **Testing on Hugging Face:**
   - You can test out the system on Hugging Face at the following URL: [https://huggingface.co/spaces/AOML-Project/ML-Screener](https://huggingface.co/spaces/AOML-Project/ML-Screener)

2. **Installation:**
   - Clone the repository and install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Running the System:**
   - Execute the main script:
     ```bash
     python app.py
     ```

---

## Future Enhancements

- **Model Refinement:** Explore additional machine learning models and deep learning techniques to further boost prediction accuracy.
- **Feature Expansion:** Integrate more advanced technical indicators and alternative data sources.
- **User Interface Improvements:** Develop an interactive web dashboard for real-time monitoring and visualization.
- **Backtesting Framework:** Implement backtesting capabilities to evaluate historical performance and optimize the system.

---

## Acknowledgements

- **Upstox API:** For providing reliable market data.
- **TA Library:** For the comprehensive suite of technical analysis tools.
- **mplfinance:** For enabling detailed and informative candlestick chart visualizations.
- **FinBERT:** For advanced financial sentiment analysis.
- **GNews Library:** For retrieving up-to-date news headlines.

---

This project represents a comprehensive approach to predicting stock reversals by harnessing multimodal data sources and ensemble learning techniques. Contributions, feedback, and further enhancements are welcome. Happy trading!
