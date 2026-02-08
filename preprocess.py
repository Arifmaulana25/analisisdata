"""
Script Python untuk preprocessing data Excel
"""
# ... import yang sudah ada ...
import sys
import json
import mysql.connector
from mysql.connector import Error
import os
import pandas as pd
from merge_duplicates import merge_duplicates, detect_columns  # <-- TAMBAHKAN BARIS INI
def connect_db():
    """Koneksi ke database MySQL"""
    try:
        connection = mysql.connector.connect(
            host=os.environ.get('DB_HOST', 'localhost'),
            database=os.environ.get('DB_NAME', 'analisis_data'),
            user=os.environ.get('DB_USER', 'root'),
            password=os.environ.get('DB_PASS', '')
        )
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        sys.exit(1)

def preprocess_excel(file_path, file_id):
    """Preprocess file Excel dan simpan ke database"""
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
        
        # 5. GABUNG BARIS DUPLIKAT BERDASARKAN (Asal, Tujuan, Komoditas), JUMLAHKAN VOLUME
        asal_col, tujuan_col, komoditas_col, volume_col = detect_columns(df)
        df_merged = merge_duplicates(df)

        # di helper, hasil penjumlahan disimpan di kolom 'Volume_Sum'
        # supaya tetap cocok dengan kolom aslinya (mis. 'Volume'), kita kembalikan nama kolomnya
        if 'Volume_Sum' in df_merged.columns:
            df_merged = df_merged.rename(columns={'Volume_Sum': volume_col})

        # pakai dataframe yang sudah digabung
        df = df_merged.reset_index(drop=True)
        
        # Simpan ke database
        connection = connect_db()
        cursor = connection.cursor()
        
        
        # Hapus data preprocessing sebelumnya jika ada
        cursor.execute("DELETE FROM preprocessed_data WHERE upload_file_id = %s", (file_id,))
        
        # Simpan setiap baris
        for idx, row in df.iterrows():
            row_data = {}
            for col in df.columns:
                row_data[col] = str(row[col]) if pd.notna(row[col]) else ''
            
            # Simpan sebagai JSON
            features_json = json.dumps(row_data, ensure_ascii=False)
            
            cursor.execute("""
                INSERT INTO preprocessed_data (upload_file_id, kolom_nama, kolom_nilai, baris_index)
                VALUES (%s, %s, %s, %s)
            """, (file_id, 'all_features', features_json, idx))
        
        connection.commit()
        
        # Update jumlah baris di upload_files
        cursor.execute("""
            UPDATE upload_files 
            SET jumlah_baris = %s 
            WHERE id = %s
        """, (len(df), file_id))
        
        connection.commit()
        
        cursor.close()
        connection.close()
        
        print(f"Preprocessing berhasil. {len(df)} baris diproses.")
        return True
        
    except Exception as e:
        print(f"Error preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py <file_path> <file_id>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    file_id = int(sys.argv[2])
    
    if not os.path.exists(file_path):
        print(f"File tidak ditemukan: {file_path}")
        sys.exit(1)
    
    success = preprocess_excel(file_path, file_id)
    sys.exit(0 if success else 1)


