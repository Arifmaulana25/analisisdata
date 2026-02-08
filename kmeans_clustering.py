"""
Script Python untuk K-Means Clustering
"""
import sys
import pandas as pd
import numpy as np
import json
import mysql.connector
from mysql.connector import Error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os


def connect_db():
    """Koneksi ke database MySQL menggunakan environment variables jika tersedia"""
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

def load_preprocessed_data(file_id):
    """
    Load data dari tabel preprocessed_data (hasil preprocess.py),
    jadi strukturnya sama seperti yang dipakai di view_data.php.
    """
    try:
        # Koneksi ke DB
        connection = connect_db()
        cursor = connection.cursor(dictionary=True)

        # Ambil semua baris JSON hasil preprocess
        cursor.execute("""
            SELECT kolom_nilai
            FROM preprocessed_data
            WHERE upload_file_id = %s
            ORDER BY baris_index
        """, (file_id,))
        rows = cursor.fetchall()

        cursor.close()
        connection.close()

        if not rows:
            raise ValueError("Data preprocessed_data kosong untuk file_id=%s. Jalankan preprocess dulu." % file_id)

        # Parse JSON ke list of dict
        records = []
        for r in rows:
            try:
                data = json.loads(r["kolom_nilai"])
                if isinstance(data, dict):
                    records.append(data)
            except Exception:
                # kalau 1 baris rusak JSON-nya, lewati saja
                continue

        if not records:
            raise ValueError("Gagal parse JSON dari preprocessed_data untuk file_id=%s" % file_id)

        # Buat DataFrame dari data yang sudah dibersihkan (tanpa duplikat)
        df = pd.DataFrame(records)
        df = df.dropna(how="all")

        # Pilih kolom numerik untuk clustering (misalnya Volume yang sudah dijumlah)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            raise ValueError("Tidak ada kolom numerik yang ditemukan untuk clustering.")

        df_numeric = df[numeric_cols].copy()
        df_numeric = df_numeric.fillna(df_numeric.median())

        return df_numeric, df, numeric_cols

    except Exception as e:
        print(f"Error loading data from preprocessed_data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

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
            distances.append(distance)
        
        return {
            'labels': cluster_labels,
            'centroids': kmeans.cluster_centers_,
            'scaler': scaler,
            'silhouette_score': silhouette_avg,
            'distances': distances,
            'inertia': kmeans.inertia_
        }
    except Exception as e:
        print(f"Error performing clustering: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def save_clustering_results(clustering_id, file_id, df_original, df_numeric, numeric_cols, clustering_result):
    """Simpan hasil clustering ke database"""
    try:
        connection = connect_db()
        cursor = connection.cursor()
        
        # Hapus data sebelumnya jika ada
        cursor.execute("DELETE FROM cluster_details WHERE clustering_result_id = %s", (clustering_id,))
        cursor.execute("DELETE FROM cluster_centroids WHERE clustering_result_id = %s", (clustering_id,))
        
        # Simpan detail cluster untuk setiap data point
        labels = clustering_result['labels']
        distances = clustering_result['distances']
        
        for idx in range(len(df_original)):
            # Ambil features asli untuk data point ini
            features = {}
            for col in df_original.columns:
                value = df_original.iloc[idx][col]
                features[col] = str(value) if pd.notna(value) else ''
            
            features_json = json.dumps(features, ensure_ascii=False)
            
            cursor.execute("""
                INSERT INTO cluster_details 
                (clustering_result_id, data_index, cluster_label, features_json, distance_to_centroid)
                VALUES (%s, %s, %s, %s, %s)
            """, (clustering_id, idx, int(labels[idx]), features_json, float(distances[idx])))
        
        # Simpan centroid untuk setiap cluster
        centroids = clustering_result['centroids']
        scaler = clustering_result['scaler']
        
        for cluster_idx in range(len(centroids)):
            # Transform centroid kembali ke skala asli
            centroid_scaled = centroids[cluster_idx].reshape(1, -1)
            centroid_original = scaler.inverse_transform(centroid_scaled)[0]
            
            # Buat dictionary centroid
            centroid_dict = {}
            for i, col in enumerate(numeric_cols):
                centroid_dict[col] = float(centroid_original[i])
            
            centroid_json = json.dumps(centroid_dict, ensure_ascii=False)
            
            cursor.execute("""
                INSERT INTO cluster_centroids 
                (clustering_result_id, cluster_label, centroid_values)
                VALUES (%s, %s, %s)
            """, (clustering_id, cluster_idx, centroid_json))
        
        # Update status clustering menjadi completed
        cursor.execute("""
            UPDATE clustering_results 
            SET status = 'completed', completed_at = NOW() 
            WHERE id = %s
        """, (clustering_id,))
        
        # Update status file menjadi processed
        cursor.execute("""
            UPDATE upload_files 
            SET status = 'processed' 
            WHERE id = %s
        """, (file_id,))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        print(f"Clustering berhasil. {len(df_original)} data point diproses.")
        print(f"Silhouette Score: {clustering_result['silhouette_score']:.4f}")
        print(f"Inertia: {clustering_result['inertia']:.4f}")
        return True
        
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Update status menjadi error
        try:
            connection = connect_db()
            cursor = connection.cursor()
            cursor.execute("UPDATE clustering_results SET status = 'error' WHERE id = %s", (clustering_id,))
            connection.commit()
            cursor.close()
            connection.close()
        except:
            pass
        
        return False

def main():
    if len(sys.argv) < 5:
        print("Usage: python kmeans_clustering.py <file_path> <clustering_id> <file_id> <n_clusters>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    clustering_id = int(sys.argv[2])
    file_id = int(sys.argv[3])
    n_clusters = int(sys.argv[4])
    
    if not os.path.exists(file_path):
        print(f"File tidak ditemukan: {file_path}")
        sys.exit(1)
    
    if n_clusters < 2 or n_clusters > 10:
        print("Jumlah cluster harus antara 2-10")
        sys.exit(1)
    
    # Load data
    print("Loading data...")
    df_numeric, df_original, numeric_cols = load_preprocessed_data(file_path,file_id)
    
    if df_numeric is None:
        print("Gagal memuat data")
        sys.exit(1)
    
    print(f"Data loaded: {len(df_numeric)} rows, {len(numeric_cols)} numeric columns")
    print(f"Numeric columns: {', '.join(numeric_cols)}")
    
    # Perform clustering
    print(f"Performing K-Means clustering with {n_clusters} clusters...")
    clustering_result = perform_kmeans_clustering(df_numeric, n_clusters)
    
    if clustering_result is None:
        print("Gagal melakukan clustering")
        sys.exit(1)
    
    # Save results
    print("Saving results to database...")
    success = save_clustering_results(
        clustering_id, file_id, df_original, df_numeric, 
        numeric_cols, clustering_result
    )
    
    if success:
        print("Clustering completed successfully!")
        sys.exit(0)
    else:
        print("Failed to save clustering results")
        sys.exit(1)

if __name__ == "__main__":
    main()

