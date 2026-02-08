"""
Script Python untuk Prediksi Moving Average dari Database
Mendukung prediksi berlanjut (menggunakan hasil prediksi sebelumnya)
"""
import sys
import json
import mysql.connector
from mysql.connector import Error
from collections import defaultdict
import os

def connect_db():
    try:
        return mysql.connector.connect(
            host=os.environ.get('DB_HOST', 'localhost'),
            database=os.environ.get('DB_NAME', 'analisis_data'),
            user=os.environ.get('DB_USER', 'root'),
            password=os.environ.get('DB_PASS', '')
        )
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        sys.exit(1)

def get_history_data(tipe_file, bulan_prediksi, tahun_prediksi, window_size):
    """Ambil data history untuk prediksi (dari data real + hasil prediksi sebelumnya)"""
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)
    
    # Hitung bulan-bulan history yang diperlukan
    required_months = []
    for i in range(1, window_size + 1):
        bulan_check = bulan_prediksi - i
        tahun_check = tahun_prediksi
        
        while bulan_check <= 0:
            bulan_check += 12
            tahun_check -= 1
        
        required_months.append({'bulan': bulan_check, 'tahun': tahun_check})
    
    # Ambil data dari preprocessed_data untuk setiap bulan (data real)
    all_data = {}
    prediction_data = {}  # Untuk menyimpan data dari hasil prediksi
    
    for req in required_months:
        month_key = f"{req['bulan']}_{req['tahun']}"
        
        # Cek apakah ada data real (upload_files dengan status processed)
        cursor.execute("""
            SELECT uf.id, pd.kolom_nilai
            FROM upload_files uf
            JOIN preprocessed_data pd ON uf.id = pd.upload_file_id
            WHERE uf.tipe_file = %s
            AND uf.bulan = %s
            AND uf.tahun = %s
            AND uf.status = 'processed'
            ORDER BY pd.baris_index
        """, (tipe_file, req['bulan'], req['tahun']))
        
        rows = cursor.fetchall()
        
        if rows:
            # Ada data real
            all_data[month_key] = rows
        else:
            # Cek apakah ada hasil prediksi untuk bulan ini
            cursor.execute("""
                SELECT pd.asal, pd.tujuan, pd.komoditas, pd.volume_predicted, pd.history_months
                FROM prediction_results pr
                JOIN prediction_details pd ON pr.id = pd.prediction_result_id
                WHERE pr.tipe_file = %s
                AND pr.bulan_prediksi = %s
                AND pr.tahun_prediksi = %s
                AND pr.status = 'completed'
            """, (tipe_file, req['bulan'], req['tahun']))
            
            pred_rows = cursor.fetchall()
            if pred_rows:
                # Ada hasil prediksi, simpan untuk diproses nanti
                prediction_data[month_key] = pred_rows
            else:
                # Tidak ada data real maupun prediksi
                all_data[month_key] = []
    
    cursor.close()
    conn.close()
    
    return all_data, required_months, prediction_data

def convert_prediction_to_features(prediction_data):
    """Convert hasil prediksi menjadi format seperti preprocessed_data (JSON features)"""
    converted_data = {}
    
    for month_key, pred_rows in prediction_data.items():
        converted_rows = []
        print(f"Converting prediction data for {month_key}: {len(pred_rows)} rows")
        
        for pred in pred_rows:
            try:
                # Hanya perlu komoditas dan volume, asal dan tujuan tidak diperlukan
                komoditas = pred.get('komoditas', '') or ''
                volume = pred.get('volume_predicted', 0) or 0
                
                if not komoditas or volume <= 0:
                    print(f"Warning: Skipping invalid prediction row: komoditas={komoditas}, volume={volume}")
                    continue
                
                # Buat JSON features dari hasil prediksi (tanpa asal dan tujuan)
                features = {
                    'Komoditas': komoditas,
                    'Volume': float(volume)
                }
                
                # Coba tambahkan volume dengan key lain juga (untuk kompatibilitas)
                features['volumeP8'] = float(volume)
                features['Jumlah'] = float(volume)
                
                converted_rows.append({
                    'id': None,
                    'kolom_nilai': json.dumps(features, ensure_ascii=False)
                })
            except Exception as e:
                print(f"Error converting prediction row: {e}")
                continue
        
        converted_data[month_key] = converted_rows
        print(f"Converted {len(converted_rows)} rows for {month_key}")
    
    return converted_data

def extract_combination_data(all_data):
    """Ekstrak data per kombinasi (hanya Komoditas, tanpa Asal dan Tujuan)"""
    combinations = defaultdict(lambda: defaultdict(list))
    
    komoditas_keys = ['Komoditas', 'komoditas', 'Commodity', 'commodity', 'Nama Tercetak', 'nama tercetak']
    volume_keys = ['volumeP8', 'Volume', 'volume', 'Jumlah', 'jumlah', 'Harga Barang (Rp)']
    
    for month_key, rows in all_data.items():
        for row in rows:
            try:
                features = json.loads(row['kolom_nilai'])
                if not isinstance(features, dict):
                    continue
                
                komoditas = ''
                volume = 0
                
                for key, value in features.items():
                    if key in komoditas_keys and not komoditas:
                        komoditas = str(value).strip()
                    if key in volume_keys and not volume:
                        try:
                            volume = float(value)
                        except:
                            pass
                
                if not volume:
                    for val in features.values():
                        try:
                            if isinstance(val, (int, float)) and val > 0:
                                volume = float(val)
                                break
                        except:
                            pass
                
                # Hanya perlu komoditas dan volume, tidak perlu asal dan tujuan
                if komoditas and volume > 0:
                    key = komoditas  # Key hanya komoditas saja
                    combinations[key][month_key].append(volume)
            except:
                continue
    
    return combinations

def calculate_moving_average(volumes_list, window_size):
    """Hitung moving average dari list volume"""
    if len(volumes_list) < window_size:
        return None
    
    # Ambil window_size terakhir
    recent_volumes = volumes_list[-window_size:]
    return sum(recent_volumes) / len(recent_volumes)

def predict_combinations(combinations, required_months, window_size):
    """Prediksi volume untuk kombinasi yang konsisten (hanya berdasarkan komoditas)"""
    predictions = []
    
    for key, month_data in combinations.items():
        komoditas = key  # Key hanya komoditas saja
        
        # Cek apakah kombinasi ini ada di semua bulan required
        month_keys = [f"{m['bulan']}_{m['tahun']}" for m in required_months]
        if not all(mk in month_data for mk in month_keys):
            continue
        
        # Kumpulkan volume per bulan (urut)
        volumes_sequence = []
        for mk in month_keys:
            volumes_sequence.extend(month_data[mk])
        
        # Hitung moving average
        ma_value = calculate_moving_average(volumes_sequence, window_size)
        if ma_value is None:
            continue
        
        # Simpan history untuk referensi
        history = {}
        for mk in month_keys:
            history[mk] = sum(month_data[mk])
        
        predictions.append({
            'asal': '',  # Kosongkan asal dan tujuan
            'tujuan': '',
            'komoditas': komoditas,
            'volume_predicted': round(ma_value, 2),
            'history': history
        })
    
    return predictions

def save_predictions(prediction_id, predictions):
    """Simpan hasil prediksi ke database"""
    conn = connect_db()
    cursor = conn.cursor()
    
    try:
        # Hapus detail lama jika ada
        cursor.execute("DELETE FROM prediction_details WHERE prediction_result_id = %s", (prediction_id,))
        
        # Simpan setiap prediksi
        for pred in predictions:
            history_json = json.dumps(pred['history'], ensure_ascii=False)
            cursor.execute("""
                INSERT INTO prediction_details 
                (prediction_result_id, asal, tujuan, komoditas, volume_predicted, history_months)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (prediction_id, pred['asal'], pred['tujuan'], pred['komoditas'], 
                  pred['volume_predicted'], history_json))
        
        # Update status dan total
        cursor.execute("""
            UPDATE prediction_results 
            SET status = 'completed', 
                total_predicted = %s,
                completed_at = NOW()
            WHERE id = %s
        """, (len(predictions), prediction_id))
        
        conn.commit()
        print(f"Prediksi berhasil. {len(predictions)} kombinasi diprediksi.")
        return True
    except Exception as e:
        conn.rollback()
        cursor.execute("UPDATE prediction_results SET status = 'error' WHERE id = %s", (prediction_id,))
        conn.commit()
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cursor.close()
        conn.close()

def main():
    if len(sys.argv) < 6:
        print("Usage: python prediction_ma.py <tipe_file> <bulan> <tahun> <window_size> <prediction_id>")
        sys.exit(1)
    
    tipe_file = sys.argv[1]
    bulan_prediksi = int(sys.argv[2])
    tahun_prediksi = int(sys.argv[3])
    window_size = int(sys.argv[4])
    prediction_id = int(sys.argv[5])
    
    # Ambil data history (real + prediksi)
    all_data, required_months, prediction_data = get_history_data(
        tipe_file, bulan_prediksi, tahun_prediksi, window_size
    )
    
    # Convert hasil prediksi ke format features
    converted_prediction = convert_prediction_to_features(prediction_data)
    
    # Gabungkan data real dengan hasil prediksi
    for month_key, pred_rows in converted_prediction.items():
        if month_key not in all_data:
            all_data[month_key] = pred_rows
        # Jika ada data real, prioritaskan data real (tapi seharusnya tidak terjadi)
    
    # Cek apakah semua bulan required punya data
    month_keys = [f"{m['bulan']}_{m['tahun']}" for m in required_months]
    missing_months = [mk for mk in month_keys if mk not in all_data or len(all_data[mk]) == 0]
    
    if missing_months:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("UPDATE prediction_results SET status = 'error' WHERE id = %s", (prediction_id,))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Error: Data tidak lengkap untuk bulan: {', '.join(missing_months)}")
        sys.exit(1)
    
    # Debug: Cetak informasi data yang tersedia
    month_keys = [f"{m['bulan']}_{m['tahun']}" for m in required_months]
    print(f"\n=== DEBUG INFO ===")
    print(f"Bulan yang diperlukan: {', '.join(month_keys)}")
    for mk in month_keys:
        count = len(all_data.get(mk, []))
        data_type = "real" if mk not in converted_prediction else "prediction"
        print(f"  {mk}: {count} rows ({data_type})")
    
    # Ekstrak kombinasi
    combinations = extract_combination_data(all_data)
    
    # Debug: Cetak informasi kombinasi
    print(f"\nTotal kombinasi ditemukan: {len(combinations)}")
    
    # Debug: Hitung kombinasi yang konsisten
    consistent_count = 0
    inconsistent_examples = []
    for key, month_data in combinations.items():
        if all(mk in month_data for mk in month_keys):
            consistent_count += 1
        else:
            if len(inconsistent_examples) < 5:
                missing_months = [mk for mk in month_keys if mk not in month_data]
                inconsistent_examples.append(f"{key} (missing: {', '.join(missing_months)})")
    
    print(f"Kombinasi konsisten di semua bulan: {consistent_count}")
    if inconsistent_examples:
        print(f"Contoh kombinasi tidak konsisten (max 5):")
        for ex in inconsistent_examples:
            print(f"  - {ex}")
    print("==================\n")
    
    # Prediksi
    predictions = predict_combinations(combinations, required_months, window_size)
    
    # Simpan ke database
    if predictions:
        save_predictions(prediction_id, predictions)
    else:
        conn = connect_db()
        cursor = conn.cursor()
        
        # Simpan error message yang lebih detail
        error_msg = f"Tidak ada kombinasi yang bisa diprediksi. Total kombinasi: {len(combinations)}, Kombinasi konsisten: {consistent_count}"
        cursor.execute("UPDATE prediction_results SET status = 'error' WHERE id = %s", (prediction_id,))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Error: {error_msg}")
        print(f"Bulan yang diperlukan: {', '.join(month_keys)}")
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()