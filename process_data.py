"""
Script Python Terpadu untuk:
- Preprocessing
- K-Means Clustering
- Moving Average
Hasil dikembalikan dalam format JSON
"""
import sys
import pandas as pd
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
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
                    df[col].fillna(median_val, inplace=True)
                else:
                    df[col].fillna(0, inplace=True)
            else:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val[0], inplace=True)
                else:
                    df[col].fillna('', inplace=True)
        
        return df
    except Exception as e:
        return None

def perform_kmeans_clustering(df_numeric, n_clusters):
    """Lakukan K-Means clustering"""
    try:
        # Standardisasi data
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_numeric)
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
        cluster_labels = kmeans.fit_predict(df_scaled)
        
        # Hitung silhouette score
        silhouette_avg = silhouette_score(df_scaled, cluster_labels)
        
        # Hitung jarak ke centroid untuk setiap data point
        distances = []
        for i, point in enumerate(df_scaled):
            centroid = kmeans.cluster_centers_[cluster_labels[i]]
            distance = np.linalg.norm(point - centroid)
            distances.append(float(distance))
        
        # Transform centroid kembali ke skala asli
        centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
        
        return {
            'success': True,
            'labels': cluster_labels.tolist(),
            'centroids': centroids_original.tolist(),
            'silhouette_score': float(silhouette_avg),
            'distances': distances,
            'inertia': float(kmeans.inertia_),
            'n_clusters': n_clusters
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

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
    if len(sys.argv) < 4:
        result = {
            'success': False,
            'error': 'Usage: python process_data.py <file_path> <n_clusters> <window_size>'
        }
        print(json.dumps(result))
        sys.exit(1)
    
    file_path = sys.argv[1]
    n_clusters = int(sys.argv[2])
    window_size = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    if not os.path.exists(file_path):
        result = {
            'success': False,
            'error': f'File tidak ditemukan: {file_path}'
        }
        print(json.dumps(result))
        sys.exit(1)
    
    if n_clusters < 2 or n_clusters > 10:
        result = {
            'success': False,
            'error': 'Jumlah cluster harus antara 2-10'
        }
        print(json.dumps(result))
        sys.exit(1)
    
    # 1. Preprocessing
    try:
        df = preprocess_data(file_path)
        if df is None or len(df) == 0:
            result = {
                'success': False,
                'error': 'Gagal memuat atau memproses file Excel'
            }
            print(json.dumps(result))
            sys.exit(1)
    except Exception as e:
        result = {
            'success': False,
            'error': f'Error preprocessing: {str(e)}'
        }
        print(json.dumps(result))
        sys.exit(1)
    
    # Pilih hanya kolom numerik untuk clustering
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    original_numeric_cols = numeric_cols.copy()
    encoded_cols = []
    
    # Jika tidak ada kolom numerik, coba konversi kolom kategorikal menjadi numerik
    if len(numeric_cols) == 0:
        # Cari kolom yang bisa dikonversi menjadi numerik
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Coba konversi kolom yang terlihat seperti numerik (string angka)
        converted_cols = []
        for col in categorical_cols:
            try:
                # Coba konversi ke numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].notna().sum() > len(df) * 0.5:  # Jika lebih dari 50% bisa dikonversi
                    numeric_cols.append(col)
                    original_numeric_cols.append(col)
                    converted_cols.append(col)
            except:
                pass
        
        # Jika masih tidak ada, gunakan Label Encoding untuk kolom kategorikal
        if len(numeric_cols) == 0 and len(categorical_cols) > 0:
            from sklearn.preprocessing import LabelEncoder
            
            # Pilih kolom kategorikal dengan variasi yang tidak terlalu banyak (maksimal 50 unique values)
            suitable_cols = []
            for col in categorical_cols[:5]:  # Maksimal 5 kolom pertama
                unique_count = df[col].nunique()
                if 2 <= unique_count <= 50:  # Minimal 2 kategori, maksimal 50
                    suitable_cols.append(col)
            
            if len(suitable_cols) > 0:
                # Buat dataframe numerik dari encoding
                df_encoded = pd.DataFrame()
                for col in suitable_cols:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df[col].astype(str))
                    numeric_cols.append(col)
                
                # Update df dengan nilai encoded
                for col in suitable_cols:
                    le = LabelEncoder()
                    encoded_col_name = col + '_encoded'
                    df[encoded_col_name] = le.fit_transform(df[col].astype(str))
                    numeric_cols.append(encoded_col_name)
                    encoded_cols.append(encoded_col_name)
        
        # Jika masih tidak ada kolom numerik, beri error dengan informasi detail
        if len(numeric_cols) == 0:
            all_cols = df.columns.tolist()
            col_types = {col: str(df[col].dtype) for col in all_cols}
            
            result = {
                'success': False,
                'error': f'Tidak ada kolom numerik yang ditemukan untuk clustering.\n\n' +
                        f'Kolom yang ditemukan ({len(all_cols)} kolom): {", ".join(all_cols[:10])}' +
                        (f'... (dan {len(all_cols)-10} kolom lainnya)' if len(all_cols) > 10 else '') +
                        f'\n\nTipe data: {", ".join([f"{col} ({col_types[col]})" for col in all_cols[:5]])}' +
                        (f'...' if len(all_cols) > 5 else '') +
                        f'\n\nSaran: Pastikan file Excel memiliki minimal 1 kolom dengan data numerik (angka).'
            }
            print(json.dumps(result, ensure_ascii=False))
            sys.exit(1)
    
    # Pastikan original_numeric_cols dan encoded_cols sudah terdefinisi
    if 'original_numeric_cols' not in locals():
        original_numeric_cols = [col for col in numeric_cols if not col.endswith('_encoded')]
    if 'encoded_cols' not in locals():
        encoded_cols = [col for col in numeric_cols if col.endswith('_encoded')]
    
    # Buat dataframe untuk clustering
    if encoded_cols:
        # Jika ada encoded columns, gunakan yang encoded
        df_numeric = df[encoded_cols].copy()
    else:
        df_numeric = df[original_numeric_cols].copy()
    
    # Fill missing values
    for col in df_numeric.columns:
        if df_numeric[col].dtype in [np.int64, np.float64]:
            median_val = df_numeric[col].median()
            if pd.notna(median_val):
                df_numeric[col].fillna(median_val, inplace=True)
            else:
                df_numeric[col].fillna(0, inplace=True)
        else:
            df_numeric[col].fillna(0, inplace=True)
    
    # 2. K-Means Clustering
    clustering_result = perform_kmeans_clustering(df_numeric, n_clusters)
    
    if not clustering_result['success']:
        result = {
            'success': False,
            'error': f"Gagal melakukan clustering: {clustering_result.get('error', 'Unknown error')}"
        }
        print(json.dumps(result))
        sys.exit(1)
    
    # 3. Moving Average
    ma_result = calculate_moving_average(df_numeric, window_size)
    
    if not ma_result['success']:
        result = {
            'success': False,
            'error': f"Gagal menghitung moving average: {ma_result.get('error', 'Unknown error')}"
        }
        print(json.dumps(result))
        sys.exit(1)
    
    # 4. Siapkan data untuk dikembalikan
    # Simpan data asli dengan features
    data_points = []
    for idx in range(len(df)):
        features = {}
        for col in df.columns:
            value = df.iloc[idx][col]
            if pd.notna(value):
                if isinstance(value, (int, float)):
                    features[col] = float(value)
                else:
                    features[col] = str(value)
            else:
                features[col] = None
        
        data_points.append({
            'index': int(idx),
            'features': features,
            'cluster_label': int(clustering_result['labels'][idx]),
            'distance_to_centroid': clustering_result['distances'][idx]
        })
    
    # Siapkan centroid
    centroids = []
    clustering_cols = encoded_cols if encoded_cols else original_numeric_cols
    for i, centroid_values in enumerate(clustering_result['centroids']):
        centroid_dict = {}
        for j, col in enumerate(clustering_cols):
            centroid_dict[col] = float(centroid_values[j])
        centroids.append({
            'cluster_label': int(i),
            'values': centroid_dict
        })
    
    # Hasil akhir dalam format JSON
    result = {
        'success': True,
        'data': {
            'total_rows': int(len(df)),
            'numeric_columns': clustering_cols,
            'all_columns': df.columns.tolist(),
            'used_encoding': len(encoded_cols) > 0,
            'data_points': data_points,
            'clustering': {
                'n_clusters': clustering_result['n_clusters'],
                'silhouette_score': clustering_result['silhouette_score'],
                'inertia': clustering_result['inertia'],
                'centroids': centroids,
                'cluster_distribution': {}
            },
            'moving_average': {
                'window': ma_result['window'],
                'values': ma_result['moving_averages']
            }
        }
    }
    
    # Hitung distribusi cluster
    from collections import Counter
    cluster_dist = Counter(clustering_result['labels'])
    result['data']['clustering']['cluster_distribution'] = {
        str(k): int(v) for k, v in cluster_dist.items()
    }
    
    # Output JSON (tanpa indent untuk mengurangi ukuran output)
    try:
        # Gunakan compact JSON untuk mengurangi ukuran output
        json_output = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
        # Pastikan output langsung ke stdout tanpa buffering
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
        # Tangkap semua error yang tidak tertangkap
        error_result = {
            'success': False,
            'error': f'Unexpected error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}'
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)

