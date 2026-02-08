"""
Script Python untuk Clustering saja (tanpa Moving Average)
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
from merge_duplicates import merge_duplicates, detect_columns

def preprocess_data(file_path):
    """Preprocess file Excel dan gabungkan data duplikat (Asal, Tujuan, Komoditas, sum Volume)"""
    try:
        df = pd.read_excel(file_path)
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')
        df = df.reset_index(drop=True)
        
        # Isi nilai kosong
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
        
        # DETEKSI kolom Asal, Tujuan, Komoditas, Volume
        asal_col, tujuan_col, komoditas_col, volume_col = detect_columns(df)

        # GABUNG DUPLIKAT: (Asal, Tujuan, Komoditas) dengan penjumlahan Volume
        df_merged = merge_duplicates(df)

        # rename Volume_Sum kembali ke nama kolom volume asli
        if 'Volume_Sum' in df_merged.columns:
            df_merged = df_merged.rename(columns={'Volume_Sum': volume_col})

        df = df_merged.reset_index(drop=True)

        return df
    except Exception:
        traceback.print_exc()
        return None

def perform_kmeans_clustering(df_numeric, n_clusters, init='k-means++'):
    """Lakukan K-Means clustering"""
    try:
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_numeric)
        
        # Validasi init method
        if init not in ['k-means++', 'random']:
            init = 'k-means++'
        
        kmeans = KMeans(n_clusters=n_clusters, init=init, random_state=42, n_init=10, max_iter=300)
        cluster_labels = kmeans.fit_predict(df_scaled)
        
        silhouette_avg = silhouette_score(df_scaled, cluster_labels)
        
        distances = []
        for i, point in enumerate(df_scaled):
            centroid = kmeans.cluster_centers_[cluster_labels[i]]
            distance = np.linalg.norm(point - centroid)
            distances.append(float(distance))
        
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

def main():
    if len(sys.argv) < 3:
        result = {
            'success': False,
            'error': 'Usage: python clustering_only.py <file_path> <n_clusters> [init_method]'
        }
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)
    
    file_path = sys.argv[1]
    n_clusters = int(sys.argv[2])
    init_method = sys.argv[3] if len(sys.argv) > 3 else 'k-means++'
    
    if not os.path.exists(file_path):
        result = {
            'success': False,
            'error': f'File tidak ditemukan: {file_path}'
        }
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)
    
    if n_clusters < 2 or n_clusters > 10:
        result = {
            'success': False,
            'error': 'Jumlah cluster harus antara 2-10'
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
    
    # Pilih kolom numerik
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    original_numeric_cols = numeric_cols.copy()
    encoded_cols = []
    
    if len(numeric_cols) == 0:
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        for col in categorical_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].notna().sum() > len(df) * 0.5:
                    numeric_cols.append(col)
                    original_numeric_cols.append(col)
            except:
                pass
        
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
            all_cols = df.columns.tolist()
            result = {
                'success': False,
                'error': f'Tidak ada kolom numerik yang ditemukan untuk clustering.\n\nKolom yang ditemukan: {", ".join(all_cols[:10])}'
            }
            print(json.dumps(result, ensure_ascii=False))
            sys.exit(1)
    
    if 'original_numeric_cols' not in locals():
        original_numeric_cols = [col for col in numeric_cols if not col.endswith('_encoded')]
    if 'encoded_cols' not in locals():
        encoded_cols = [col for col in numeric_cols if col.endswith('_encoded')]
    
    if encoded_cols:
        df_numeric = df[encoded_cols].copy()
    else:
        df_numeric = df[original_numeric_cols].copy()
    
    for col in df_numeric.columns:
        if df_numeric[col].dtype in [np.int64, np.float64]:
            median_val = df_numeric[col].median()
            if pd.notna(median_val):
                df_numeric[col] = df_numeric[col].fillna(median_val)
            else:
                df_numeric[col] = df_numeric[col].fillna(0)
        else:
            df_numeric[col] = df_numeric[col].fillna(0)
    
    # 2. K-Means Clustering
    clustering_result = perform_kmeans_clustering(df_numeric, n_clusters, init_method)
    
    if not clustering_result['success']:
        result = {
            'success': False,
            'error': f"Gagal melakukan clustering: {clustering_result.get('error', 'Unknown error')}"
        }
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)
    
    # 3. Siapkan data untuk dikembalikan
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
            }
        }
    }
    
    from collections import Counter
    cluster_dist = Counter(clustering_result['labels'])
    result['data']['clustering']['cluster_distribution'] = {
        str(k): int(v) for k, v in cluster_dist.items()
    }
    
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

