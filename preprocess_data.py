import pandas as pd
import numpy as np
import os

# --- ĐỊNH NGHĨA ĐƯỜNG DẪN ---
# File đầu vào
FILE_GOLD_SILVER = 'data/Gold-Silver-GeopoliticalRisk_HistoricalData.csv'
FILE_GPRD_FEATURES = 'data/df_recent_preprocessed.csv'

# Thư mục và file đầu ra (giống như trong app.py)
OUTPUT_DIR = 'data'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'preprocessed_gold_data.csv')

def load_data():
    """
    Tải dữ liệu từ hai file CSV đầu vào.
    """
    print(f"Đang tải dữ liệu giá từ: {FILE_GOLD_SILVER}...")
    # Chỉ lấy các cột cần thiết từ file giá
    df_gold = pd.read_csv(
        FILE_GOLD_SILVER,
        usecols=['DATE', 'GOLD_PRICE', 'SILVER_PRICE'],
        parse_dates=['DATE']
    )
    df_gold = df_gold.sort_values(by='DATE', ascending=True)
    
    print(f"Đang tải dữ liệu GPRD từ: {FILE_GPRD_FEATURES}...")
    # Lấy các cột GPRD và các đặc trưng đã xử lý
    gprd_cols = [
        'date', 'N10D', 'GPRD', 'GPRD_ACT', 'GPRD_THREAT', 
        'GPRD_MA30', 'GPRD_MA7', 'GPRD_smoothed'
    ]
    df_gprd = pd.read_csv(
        FILE_GPRD_FEATURES,
        usecols=lambda c: c in gprd_cols, # Chỉ tải các cột có trong danh sách
        parse_dates=['date']
    )
    df_gprd = df_gprd.sort_values(by='date', ascending=True)

    return df_gold, df_gprd

## gop cac cot dam bao khong co du lieu null

def merge_data(df_gold, df_gprd):
    """
    Gộp hai DataFrame lại dựa trên ngày.
    """
    print("Đang gộp dữ liệu giá và dữ liệu GPRD...")
    # Gộp 'inner' để đảm bảo chỉ giữ lại những ngày có cả hai loại dữ liệu
    df_merged = pd.merge(
        df_gold, 
        df_gprd, 
        left_on='DATE', 
        right_on='date', 
        how='inner'
    )
    # Loại bỏ cột 'date' bị trùng lặp
    df_merged = df_merged.drop(columns=['date'])
    return df_merged

def feature_engineering(df):
    """
    Tạo các đặc trưng mới cho mô hình.
    """
    print("Đang tạo các đặc trưng mới (feature engineering)...")
    df_feat = df.copy()

    # Tạo các đặc trưng trễ (lags) cho giá Vàng và Bạc
    # (shift(1) để đảm bảo không dùng thông tin của ngày hiện tại để dự đoán chính nó)
    df_feat['GOLD_PRICE_lag1'] = df_feat['GOLD_PRICE'].shift(1)
    df_feat['GOLD_PRICE_lag7'] = df_feat['GOLD_PRICE'].shift(7)
    df_feat['SILVER_PRICE_lag1'] = df_feat['SILVER_PRICE'].shift(1)

    # Tạo các đặc trưng trung bình trượt (rolling means)
    df_feat['GOLD_PRICE_roll_mean_7'] = df_feat['GOLD_PRICE'].shift(1).rolling(window=7).mean()
    df_feat['GOLD_PRICE_roll_mean_30'] = df_feat['GOLD_PRICE'].shift(1).rolling(window=30).mean()
    df_feat['SILVER_PRICE_roll_mean_7'] = df_feat['SILVER_PRICE'].shift(1).rolling(window=7).mean()

    # Tỷ lệ Vàng/Bạc
    # Thêm 1e-6 để tránh lỗi chia cho 0 (mặc dù giá bạc hiếm khi bằng 0)
    df_feat['GS_RATIO'] = df_feat['GOLD_PRICE'].shift(1) / (df_feat['SILVER_PRICE'].shift(1) + 1e-6)

    # Đặc trưng về thời gian
    df_feat['month'] = df_feat['DATE'].dt.month
    df_feat['day_of_week'] = df_feat['DATE'].dt.dayofweek
    df_feat['day_of_year'] = df_feat['DATE'].dt.dayofyear
    
    return df_feat

def clean_data(df):
    """
    Loại bỏ các hàng có giá trị NaN (thường là ở đầu dữ liệu do rolling/shift).
    """
    print("Đang loại bỏ các hàng có giá trị rỗng (NaN)...")
    original_rows = len(df)
    df_cleaned = df.dropna()
    removed_rows = original_rows - len(df_cleaned)
    print(f"Đã loại bỏ {removed_rows} hàng. Dữ liệu cuối cùng có {len(df_cleaned)} hàng.")
    
    return df_cleaned

def save_data(df, path):
    """
    Lưu DataFrame cuối cùng ra file CSV.
    """
    print(f"Đang lưu dữ liệu đã xử lý vào: {path}...")
    # Tạo thư mục nếu nó không tồn tại
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Lưu file, không bao gồm index của pandas
    # app.py tải file này và tự đặt index bằng cột 'DATE'
    df.to_csv(path, index=False)
    print("Lưu file thành công!")

# --- HÀM CHÍNH ĐỂ CHẠY QUÁ TRÌNH ---
def main():
    print("===== BẮT ĐẦU QUÁ TRÌNH TIỀN XỬ LÝ DỮ LIỆU =====")
    
    try:
        df_gold, df_gprd = load_data()
        print(f"Tải xong {len(df_gold)} dòng giá và {len(df_gprd)} dòng GPRD.")
        
        df_merged = merge_data(df_gold, df_gprd)
        print(f"Dữ liệu đã gộp. Tổng số dòng (trước khi xử lý): {len(df_merged)}.")
        
        df_featured = feature_engineering(df_merged)
        
        df_final = clean_data(df_featured)
        
        save_data(df_final, OUTPUT_FILE)
        
        print("\n===== TIỀN XỬ LÝ HOÀN TẤT =====")
        print(f"File dữ liệu đã sẵn sàng tại: {OUTPUT_FILE}")
        print("\nCác cột trong file cuối cùng:")
        print(df_final.columns.tolist())
        
    except FileNotFoundError as e:
        print(f"\nLỖI: Không tìm thấy file. Vui lòng kiểm tra lại đường dẫn.")
        print(f"Chi tiết lỗi: {e}")
    except Exception as e:
        print(f"\nĐã xảy ra lỗi không mong muốn:")
        print(f"Chi tiết lỗi: {e}")

if __name__ == "__main__":
    main()