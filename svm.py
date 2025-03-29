import os
import pandas as pd
import requests
import json
import mplfinance as mpf
import streamlit as st
from datetime import datetime, timedelta
from io import StringIO
from PIL import Image
import cv2
import numpy as np
import joblib
import shutil

def svm():
   
    url = 'https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv'
    csv_file = 'artifacts/nifty_50_symbols.csv'
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/110.0 Safari/537.36'
    }
    def is_csv_up_to_date(file_path):
        if os.path.exists(file_path):
            file_date = datetime.fromtimestamp(os.path.getmtime(file_path)).date()
            current_date = datetime.now().date()
            return file_date == current_date
        return False

    try:
        if is_csv_up_to_date(csv_file):
            df = pd.read_csv(csv_file)
        else:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = StringIO(response.text)
            df = pd.read_csv(data)
            df.to_csv(csv_file, index=False)
        
    except Exception as ex:
        st.write(f"An error occurred: {ex}")
        df = pd.read_csv(csv_file)

    tickers = df['Symbol'].to_list()
    
    with open('artifacts/NSE.json', 'r') as file:
        data_json = json.load(file)

    filtered_data = [
        item for item in data_json
        if (item.get('segment') == 'NSE_EQ')
    ]
    df_json = pd.DataFrame(filtered_data)
    df_json = df_json[df_json['trading_symbol'].isin(tickers)]
    instruments_dict = df_json.set_index('trading_symbol')['instrument_key'].to_dict()
    end_date= datetime.today()
    start_date = end_date - timedelta(days=7)
    start_date=start_date.strftime("%Y-%m-%d")
    end_date=end_date.strftime("%Y-%m-%d")
    folder = "images"

    # Delete the contents of the folder if it exists
    if os.path.exists(folder):
        shutil.rmtree(folder)  # Remove the entire folder and its contents

    # Recreate the empty folder
    os.makedirs(folder, exist_ok=True)

    for key, value in instruments_dict.items():
        
        url = f"https://api.upstox.com/v2/historical-candle/{value}/day/{end_date}/{start_date}"
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/110.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers)

        if response.status_code == 200:

            try:
                data_response = response.json()
                candles = data_response['data']['candles']
                df_stock = pd.DataFrame(candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Open_Interest'])
                df_stock['Timestamp'] = pd.to_datetime(df_stock['Timestamp'])
                df_stock['Date'] = df_stock['Timestamp'].dt.strftime('%Y-%m-%d')
                df_stock = df_stock.iloc[::-1].reset_index(drop=True) 


                filename = f"{folder}/{key}_{df_stock['Date'].iloc[-2]}.png"

                plot_df = df_stock.iloc[-3:].copy()

                if 'Timestamp' in plot_df.columns:
                    plot_df.set_index('Timestamp', inplace=True)

                mc = mpf.make_marketcolors(up='green', down='red', edge='black', wick='black')
                s = mpf.make_mpf_style(marketcolors=mc)

                mpf.plot(
                    plot_df,
                    type='candle',
                    style=s,
                    title=f"{key} - {df_stock['Date'].iloc[-2]}",
                    ylabel='Price',
                    savefig=filename
                )

            except:
                print("exception")
                continue
        else:
            print(response)

    corner1 = (147, 72)    
    corner2 = (718, 72)    
    corner3 = (718, 470)   
    corner4 = (147, 470)  

    x_min = min(corner1[0], corner2[0], corner3[0], corner4[0])
    y_min = min(corner1[1], corner2[1], corner3[1], corner4[1])
    x_max = max(corner1[0], corner2[0], corner3[0], corner4[0])
    y_max = max(corner1[1], corner2[1], corner3[1], corner4[1])

    folders = ['images']

    for folder in folders:
        folder_path = os.path.join(os.getcwd(), folder)  

        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                
                image_path = os.path.join(folder_path, filename)

                try:
                    img = Image.open(image_path)
                    cropped_img = img.crop((x_min, y_min, x_max, y_max))

                    cropped_img.save(image_path)

                except Exception as e:
                    continue

    model_filename = "pickle/svm_final.pkl"
    pca_filename = "pickle/svm_pca.pkl"

    model = joblib.load(model_filename)
    pca = joblib.load(pca_filename)

    class_names = ['bearish', 'bullish']

    def preprocess_image_cpu(image_path, image_size=(224, 224)):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Image not found or unable to load.")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_size)
            img = img.astype(np.float32) / 255.0
            return img.flatten()
        except Exception as e:
            print(f"⚠️ Error processing image {image_path}: {e}")
            return None

    def predict_proba_cpu(model, pca, image_path, class_names):
        img = preprocess_image_cpu(image_path)
        if img is None:
            return None
    
        img_batch = np.array([img])
 
        img_pca = pca.transform(img_batch)

        probabilities = model.predict_proba(img_pca)[0] 
        
        prob_dict = {
            class_names[0]: float(probabilities[0]),  
            class_names[1]: float(probabilities[1])    
        }
        
        return prob_dict

    image_folder = "images"

    results = []

    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
 
        if os.path.isfile(image_path):
            probs = predict_proba_cpu(model, pca, image_path, class_names)
            
            if probs:
                results.append({
                    "Stock": image_name.split("_")[0],
                    "SVM_Bearish_Probability": probs['bearish'],
                    "SVM_Bullish_Probability": probs['bullish']
                })
            else:
                print(f"⚠️ Failed to process {image_name}")

    df_probs = pd.DataFrame(results)
    return df_probs

