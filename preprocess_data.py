import pandas as pd
import numpy as np
import os

# --- ĐỊNH NGHĨA ĐƯỜNG DẪN ---
FILE_GOLD_SILVER = 'data/Gold-Silver-GeopoliticalRisk_HistoricalData.csv'
FILE_GPRD_FEATURES = 'data/df_recent_preprocessed.csv' # File này có thể không tồn tại
OUTPUT_DIR = 'data'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'preprocessed_data.csv')

def load_data():
    """
    Tải dữ liệu từ các file CSV một cách linh hoạt.
    """
    # --- Tải dữ liệu giá Gold/Silver ---
    print(f"Đang tải dữ liệu giá từ: {FILE_GOLD_SILVER}...")
    try:
        # Đọc tất cả các cột có sẵn trong file csv trước
        df_price_cols = pd.read_csv(FILE_GOLD_SILVER, nrows=0).columns.tolist()
        
        # Xác định các cột sẽ sử dụng
        price_use_cols = ['DATE', 'GOLD_PRICE']
        if 'SILVER_PRICE' in df_price_cols:
            print("- Tìm thấy cột 'SILVER_PRICE'.")
            price_use_cols.append('SILVER_PRICE')
        else:
            print("⚠️ CẢNH BÁO: Không tìm thấy cột 'SILVER_PRICE' trong file. Các tính năng liên quan đến bạc sẽ bị bỏ qua.")

        df_price = pd.read_csv(
            FILE_GOLD_SILVER,
            usecols=price_use_cols,
            parse_dates=['DATE']
        )
        df_price = df_price.sort_values(by='DATE', ascending=True)
    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy file giá '{FILE_GOLD_SILVER}'. Không thể tiếp tục.")
        return None, None
    except Exception as e:
        print(f"❌ LỖI: Không thể đọc file giá. Lỗi: {e}")
        return None, None

    # --- Tải dữ liệu GPRD (nếu có) ---
    df_gprd = None
    if os.path.exists(FILE_GPRD_FEATURES):
        print(f"Đang tải dữ liệu GPRD từ: {FILE_GPRD_FEATURES}...")
        try:
            df_gprd_cols = pd.read_csv(FILE_GPRD_FEATURES, nrows=0).columns.tolist()
            gprd_use_cols = ['date']
            if 'GPRD' in df_gprd_cols:
                 print("- Tìm thấy cột 'GPRD'.")
                 gprd_use_cols.append('GPRD')
            else:
                print("⚠️ CẢNH BÁO: Không tìm thấy cột 'GPRD' trong file GPRD.")

            if gprd_use_cols:
                 df_gprd = pd.read_csv(
                    FILE_GPRD_FEATURES,
                    usecols=gprd_use_cols,
                    parse_dates=['date']
                )
                 df_gprd = df_gprd.sort_values(by='date', ascending=True)
        except Exception as e:
            print(f"⚠️ CẢNH BÁO: Không thể đọc file GPRD. Lỗi: {e}. Bỏ qua dữ liệu GPRD.")
            df_gprd = None
    else:
        print(f"ℹ️ INFO: File '{FILE_GPRD_FEATURES}' không tồn tại, bỏ qua dữ liệu GPRD.")

    return df_price, df_gprd

def merge_data(df_price, df_gprd):
    """
    Gộp hai DataFrame lại dựa trên ngày bằng 'left' join.
    """
    if df_gprd is None:
        print("Không có dữ liệu GPRD để gộp, sử dụng dữ liệu giá gốc.")
        return df_price
        
    print("Đang gộp dữ liệu giá và dữ liệu GPRD bằng 'left' join...")
    # Gộp 'left' để luôn giữ lại tất cả dữ liệu giá, kể cả khi không có GPRD
    df_merged = pd.merge(
        df_price, 
        df_gprd, 
        left_on='DATE', 
        right_on='date', 
        how='left'
    )
    # Loại bỏ cột 'date' bị trùng lặp nếu có
    if 'date' in df_merged.columns:
        df_merged = df_merged.drop(columns=['date'])
    return df_merged

def feature_engineering(df):
    """
    Tạo các đặc trưng mới cho mô hình một cách linh hoạt.
    """
    print("Đang tạo các đặc trưng mới (feature engineering)...")
    df_feat = df.copy()
    
    # --- Luôn tạo đặc trưng cho Vàng ---
    df_feat['GOLD_PRICE_lag1'] = df_feat['GOLD_PRICE'].shift(1)
    df_feat['GOLD_PRICE_lag7'] = df_feat['GOLD_PRICE'].shift(7)
    df_feat['GOLD_PRICE_roll_mean_7'] = df_feat['GOLD_PRICE'].shift(1).rolling(window=7).mean()
    df_feat['GOLD_PRICE_roll_mean_30'] = df_feat['GOLD_PRICE'].shift(1).rolling(window=30).mean()

    # --- Tạo đặc trưng cho Bạc (nếu có) ---
    if 'SILVER_PRICE' in df_feat.columns:
        print("- Đang tạo đặc trưng cho Giá Bạc...")
        df_feat['SILVER_PRICE_lag1'] = df_feat['SILVER_PRICE'].shift(1)
        df_feat['SILVER_PRICE_roll_mean_7'] = df_feat['SILVER_PRICE'].shift(1).rolling(window=7).mean()
        
        # Tỷ lệ Vàng/Bạc (chỉ tạo khi có cả 2)
        print("- Đang tạo đặc trưng Tỷ lệ Vàng/Bạc (GS_RATIO)...")
        # Thêm 1e-6 để tránh lỗi chia cho 0
        df_feat['GS_RATIO'] = df_feat['GOLD_PRICE'].shift(1) / (df_feat['SILVER_PRICE'].shift(1) + 1e-6)

    # --- Đặc trưng về thời gian ---
    df_feat['month'] = df_feat['DATE'].dt.month
    df_feat['day_of_week'] = df_feat['DATE'].dt.dayofweek
    df_feat['day_of_year'] = df_feat['DATE'].dt.dayofyear
    
    return df_feat

def clean_data(df):
    """
    Loại bỏ các hàng có giá trị NaN cốt lõi (sau khi shift và rolling).
    """
    print("Đang loại bỏ các hàng có giá trị rỗng (NaN) cốt lõi...")
    original_rows = len(df)
    # Chỉ loại bỏ các hàng thiếu các đặc trưng lag cơ bản của giá vàng
    df_cleaned = df.dropna(subset=['GOLD_PRICE_lag1', 'GOLD_PRICE_roll_mean_30'])
    removed_rows = original_rows - len(df_cleaned)
    print(f"Đã loại bỏ {removed_rows} hàng. Dữ liệu cuối cùng có {len(df_cleaned)} hàng.")
    
    return df_cleaned

def save_data(df, path):
    """
    Lưu DataFrame cuối cùng ra file CSV.
    """
    print(f"Đang lưu dữ liệu đã xử lý vào: {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Lưu file thành công vào '{path}'!")

# --- HÀM CHÍNH ĐỂ CHẠY QUÁ TRÌNH ---
def main():
    print("===== BẮT ĐẦU QUÁ TRÌNH TIỀN XỬ LÝ DỮ LIỆU (PHIÊN BẢN NÂNG CẤP) =====")
    
    df_price, df_gprd = load_data()
    
    if df_price is None:
        return 

    df_merged = merge_data(df_price, df_gprd)
    
    df_featured = feature_engineering(df_merged)
    
    df_final = clean_data(df_featured)
    
    if df_final.empty:
        print("❌ LỖI: Không còn dữ liệu sau khi xử lý. Vui lòng kiểm tra lại file đầu vào.")
        return

    save_data(df_final, OUTPUT_FILE)
    
    print("\n===== TIỀN XỬ LÝ HOÀN TẤT =====")
    print("\nCác cột có trong file cuối cùng:")
    print(df_final.columns.tolist())

if __name__ == "__main__":
    main()