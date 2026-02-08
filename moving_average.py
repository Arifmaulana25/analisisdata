"""
Script Python untuk Moving Average / Prediksi
Hanya menghitung moving average, tidak melakukan clustering
"""
import sys
import pandas as pd
import numpy as np
import json
import os
import traceback

def preprocess_data(file_path):
    """Preprocess file Excel"""
    try:
        # Baca file Excel
        df = pd.read_excel(file_path)
        
        # Basic preprocessing
        # 1. Hapus baris kosong
        df = df.dropna(how='all')
        
        # 2. Hapus kolom yang seluruhnya kosong
        df = df.dropna(axis=1, how='all')
        
        # 3. Reset index
        df = df.reset_index(drop=True)
        
        # 4. Fill missing values dengan median untuk numeric, mode untuk categorical
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                median_val = df[col].median()
                if pd.notna(median_val):
                    df[col] = df[col].fillna(median_val)
                else:
                    df[col] = df[col].fillna(0)
            else:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    df[col] = df[col].fillna('')
        
        return df
    except Exception as e:
        return None

def calculate_moving_average(df_numeric, window=3):
    """Hitung Moving Average untuk setiap kolom numerik"""
    try:
        ma_results = {}
        
        for col in df_numeric.columns:
            values = df_numeric[col].values
            ma = pd.Series(values).rolling(window=window, min_periods=1).mean()
            ma_results[col] = ma.tolist()
        
        return {
            'success': True,
            'moving_averages': ma_results,
            'window': window
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def main():
    if len(sys.argv) < 3:
        result = {
            'success': False,
            'error': 'Usage: python moving_average.py <file_path> <window_size>'
        }
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)
    
    file_path = sys.argv[1]
    window_size = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    if not os.path.exists(file_path):
        result = {
            'success': False,
            'error': f'File tidak ditemukan: {file_path}'
        }
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)
    
    if window_size < 2 or window_size > 10:
        result = {
            'success': False,
            'error': 'Window size harus antara 2-10'
        }
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)
    
    # 1. Preprocessing
    try:
        df = preprocess_data(file_path)
        if df is None or len(df) == 0:
            result = {
                'success': False,
                'error': 'Gagal memuat atau memproses file Excel'
            }
            print(json.dumps(result, ensure_ascii=False))
            sys.exit(1)
    except Exception as e:
        result = {
            'success': False,
            'error': f'Error preprocessing: {str(e)}'
        }
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)
    
    # Pilih hanya kolom numerik untuk moving average
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    original_numeric_cols = numeric_cols.copy()
    encoded_cols = []
    
    # Jika tidak ada kolom numerik, coba konversi
    if len(numeric_cols) == 0:
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Coba konversi kolom string angka
        for col in categorical_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].notna().sum() > len(df) * 0.5:
                    numeric_cols.append(col)
                    original_numeric_cols.append(col)
            except:
                pass
        
        # Jika masih tidak ada, gunakan Label Encoding
        if len(numeric_cols) == 0 and len(categorical_cols) > 0:
            from sklearn.preprocessing import LabelEncoder
            
            suitable_cols = []
            for col in categorical_cols[:5]:
                unique_count = df[col].nunique()
                if 2 <= unique_count <= 50:
                    suitable_cols.append(col)
            
            if len(suitable_cols) > 0:
                for col in suitable_cols:
                    le = LabelEncoder()
                    encoded_col_name = col + '_encoded'
                    df[encoded_col_name] = le.fit_transform(df[col].astype(str))
                    numeric_cols.append(encoded_col_name)
                    encoded_cols.append(encoded_col_name)
        
        if len(numeric_cols) == 0:
            result = {
                'success': False,
                'error': 'Tidak ada kolom numerik yang ditemukan untuk moving average'
            }
            print(json.dumps(result, ensure_ascii=False))
            sys.exit(1)
    
    # Buat dataframe numerik
    if encoded_cols:
        df_numeric = df[encoded_cols].copy()
    else:
        df_numeric = df[original_numeric_cols].copy()
    
    # Fill missing values
    for col in df_numeric.columns:
        if df_numeric[col].dtype in [np.int64, np.float64]:
            median_val = df_numeric[col].median()
            if pd.notna(median_val):
                df_numeric[col] = df_numeric[col].fillna(median_val)
            else:
                df_numeric[col] = df_numeric[col].fillna(0)
        else:
            df_numeric[col] = df_numeric[col].fillna(0)
    
    # 2. Moving Average
    ma_result = calculate_moving_average(df_numeric, window_size)
    
    if not ma_result['success']:
        result = {
            'success': False,
            'error': f"Gagal menghitung moving average: {ma_result.get('error', 'Unknown error')}"
        }
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)
    
    # Hasil akhir dalam format JSON
    result = {
        'success': True,
        'data': {
            'total_rows': int(len(df)),
            'numeric_columns': encoded_cols if encoded_cols else original_numeric_cols,
            'all_columns': df.columns.tolist(),
            'moving_average': {
                'window': ma_result['window'],
                'values': ma_result['moving_averages']
            }
        }
    }
    
    # Output JSON (compact)
    try:
        json_output = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
        sys.stdout.write(json_output)
        sys.stdout.flush()
        sys.exit(0)
    except Exception as e:
        error_result = {
            'success': False,
            'error': f'Error saat membuat output JSON: {str(e)}'
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_result = {
            'success': False,
            'error': f'Unexpected error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}'
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)

