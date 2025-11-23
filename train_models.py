import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib  # Để lưu model
import json    # Để lưu file config
import os
from datetime import datetime

# --- Các hằng số ---
# Đường dẫn file input (đã sửa để khớp với file bạn tải lên)
INPUT_FILE = "data/Gold-Silver-GeopoliticalRisk_HistoricalData.csv" 
# Đường dẫn file output (app.py sẽ đọc file này)
PROCESSED_FILE = "data/preprocessed_gold_data.csv" 
MODEL_DIR = "models"
DATA_DIR = "data"

# Tạo thư mục nếu chưa có
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print("Bắt đầu quá trình huấn luyện...")

# =============================================================================
# BƯỚC 1 & 2: TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
# =============================================================================
print(f"Đang tải dữ liệu từ file: {INPUT_FILE}...")
try:
    # Chỉ tải các cột cần thiết
    df = pd.read_csv(INPUT_FILE, usecols=['DATE', 'GOLD_PRICE'])
    
    # 1. Xử lý cột DATE
    df_processed = df[['DATE', 'GOLD_PRICE']].copy()
    df_processed['DATE'] = pd.to_datetime(df_processed['DATE'], errors='coerce')
    df_processed = df_processed.dropna(subset=['DATE'])
    df_processed = df_processed.sort_values('DATE')
    df_processed = df_processed.set_index('DATE') # Đặt DATE làm index

    # 2. Xử lý cột GOLD_PRICE
    df_processed['GOLD_PRICE'] = pd.to_numeric(df_processed['GOLD_PRICE'], errors='coerce')
    
    # 3. Fill các giá trị thiếu (nếu có)
    # Lấy ngày đầu tiên và ngày cuối cùng có dữ liệu
    start_date = df_processed.index.min()
    end_date = df_processed.index.max()
    
    # Tạo một dải ngày liên tục (bao gồm cả cuối tuần)
    all_days = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Reindex DataFrame để bao gồm tất cả các ngày, các ngày thiếu sẽ là NaN
    df_processed = df_processed.reindex(all_days)
    
    # Điền giá trị NaN bằng cách sử dụng giá trị hợp lệ cuối cùng (forward fill)
    # Điều này giả định giá cuối tuần/ngày lễ bằng giá ngày giao dịch trước đó
    df_processed['GOLD_PRICE'] = df_processed['GOLD_PRICE'].fillna(method='ffill')
    
    # Nếu vẫn còn NaN ở đầu (trường hợp ngày đầu tiên bị thiếu), thì fill ngược
    df_processed = df_processed.fillna(method='bfill')
    
    print("Dữ liệu đã được tải và làm sạch.")

except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file '{INPUT_FILE}'. Vui lòng đảm bảo file này ở cùng thư mục.")
    exit()
except Exception as e:
    print(f"LỖI khi tải dữ liệu: {e}")
    exit()

# =============================================================================
# BƯỚC 3: FEATURE ENGINEERING (TẠO ĐẶC TRƯNG)
# =============================================================================
print("Đang tạo các đặc trưng (feature engineering)...")
# y là biến mục tiêu (target)
y = df_processed['GOLD_PRICE']

# X là DataFrame chứa các đặc trưng (features)
X = pd.DataFrame(index=df_processed.index)

# Tạo các đặc trưng (features)
# Chúng ta shift(1) để đảm bảo không dùng dữ liệu hôm nay để dự đoán chính nó
X['GOLD_PRICE_lag1'] = y.shift(1)
X['GOLD_PRICE_lag7'] = y.shift(7)
X['GOLD_PRICE_lag30'] = y.shift(30)

X['GOLD_PRICE_roll_mean_7'] = y.shift(1).rolling(window=7).mean()
X['GOLD_PRICE_roll_mean_30'] = y.shift(1).rolling(window=30).mean()

X['GOLD_PRICE_roll_std_7'] = y.shift(1).rolling(window=7).std()
X['GOLD_PRICE_roll_std_30'] = y.shift(1).rolling(window=30).std()

# Đặc trưng thời gian
X['month'] = X.index.month
X['day_of_week'] = X.index.dayofweek
X['day_of_year'] = X.index.dayofyear

# Loại bỏ các hàng NaN được tạo ra do shift() và rolling()
# (thường là 30 hàng đầu tiên)
print(f"Số dòng ban đầu: {len(X)}")
X = X.dropna()
y = y[X.index] # Đảm bảo y và X khớp nhau
print(f"Số dòng sau khi loại bỏ NaN: {len(X)}")

# =============================================================================
# BƯỚC 4: LƯU FILE DỮ LIỆU ĐÃ XỬ LÝ (CHO app.py)
# =============================================================================
print(f"Đang lưu file tiền xử lý vào: {PROCESSED_FILE}")
# Gộp X và y lại để lưu
df_to_save = X.join(y)
# reset_index() để cột 'DATE' (đang là index) trở thành một cột bình thường
df_to_save.reset_index().rename(columns={'index': 'DATE'}).to_csv(PROCESSED_FILE, index=False)
print("- Đã lưu file tiền xử lý.")

# =============================================================================
# BƯỚC 5: CHIA DỮ LIỆU VÀ CHUẨN HÓA
# =============================================================================
print("Đang chia dữ liệu và chuẩn hóa...")
# Dành 20% cuối cùng cho tập test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lấy danh sách tên các cột đặc trưng (features)
features_columns = X.columns.tolist()

print(f"Tập huấn luyện: {X_train_scaled.shape[0]} mẫu")
print(f"Tập kiểm tra: {X_test_scaled.shape[0]} mẫu")

# =============================================================================
# BƯỚC 6: HUẤN LUYỆN MODELS
# =============================================================================
print("Đang huấn luyện mô hình Random Forest...")
model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_rf.fit(X_train_scaled, y_train)
y_pred_rf = model_rf.predict(X_test_scaled)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
print(f"- Random Forest - RMSE: {rmse_rf:.2f}, R-squared: {r2_rf:.2f}")

print("Đang huấn luyện mô hình XGBoost...")
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1)
model_xgb.fit(X_train_scaled, y_train)
y_pred_xgb = model_xgb.predict(X_test_scaled)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"- XGBoost - RMSE: {rmse_xgb:.2f}, R-squared: {r2_xgb:.2f}")

# =============================================================================
# BƯỚC 7: LƯU TẤT CẢ CÁC ARTIFACTS (CHO app.py)
# =============================================================================
print("Đang lưu tất cả các artifacts (models, scaler, json)...")

# 1. Lưu Models
best_model_name = 'Random Forest' if rmse_rf <= rmse_xgb else 'XGBoost'
best_model_obj = model_rf if best_model_name == 'Random Forest' else model_xgb
joblib.dump(model_rf, os.path.join(MODEL_DIR, 'model_rf.pkl'))
joblib.dump(model_xgb, os.path.join(MODEL_DIR, 'model_xgb.pkl'))
print(f"- Đã lưu models RF và XGB. (Mô hình tốt nhất: {best_model_name})")

# 2. Lưu Scaler
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
print("- Đã lưu Scaler.")

# 3. Lưu kết quả so sánh
model_comparison = {
    'Random Forest': {'RMSE': round(rmse_rf, 2), 'R-squared': round(r2_rf, 2)},
    'XGBoost': {'RMSE': round(rmse_xgb, 2), 'R-squared': round(r2_xgb, 2)},
    'best_model': best_model_name,
    'test_set_size': len(y_test),
    # Thêm siêu tham số của model tốt nhất
    'best_model_hyperparameters': best_model_obj.get_params()
}
with open(os.path.join(MODEL_DIR, 'model_comparison.json'), 'w') as f:
    json.dump(model_comparison, f, indent=4)
print("- Đã lưu kết quả so sánh (bao gồm cả hyperparameters).")

# 4. Lưu dữ liệu cho logic dự đoán (mô phỏng trong app.py)
# Tính xu hướng 30 ngày cuối cùng trong *toàn bộ* tập dữ liệu
trend_30_day = (y.iloc[-1] - y.iloc[-30]) / 30 
latest_data = {
    'date': y.index[-1].strftime('%Y-%m-%d'),
    'price': y.iloc[-1], # Giá trị cuối cùng
    'trend_30_day': trend_30_day
}
with open(os.path.join(MODEL_DIR, 'latest_data.json'), 'w') as f:
    json.dump(latest_data, f, indent=4)
print("- Đã lưu dữ liệu dự đoán mới nhất.")

# 5. Lưu dữ liệu cho biểu đồ so sánh trên tập Test
# Chuyển ngày tháng sang string để JSON có thể đọc được
test_chart_data = {
    'dates': [d.strftime('%Y-%m-%d') for d in y_test.index],
    'actual': y_test.tolist(),
    # LƯU Ý: Lưu dự đoán của CẢ HAI mô hình
    'predictions': {
        'Random Forest': y_pred_rf.tolist(),
        'XGBoost': y_pred_xgb.tolist()
    }
}
with open(os.path.join(MODEL_DIR, 'test_chart_data.json'), 'w') as f:
    json.dump(test_chart_data, f)
print("- Đã lưu dữ liệu biểu đồ test.")

# 6. Lưu dữ liệu cho biểu đồ sai số (residuals)
# LƯU Ý: Tính và lưu sai số cho CẢ HAI mô hình
residuals_rf = y_test - y_pred_rf
residuals_xgb = y_test - y_pred_xgb
residual_chart_data = {
    'dates': [d.strftime('%Y-%m-%d') for d in y_test.index],
    'residuals': {
        'Random Forest': residuals_rf.tolist(),
        'XGBoost': residuals_xgb.tolist()
    }
}
with open(os.path.join(MODEL_DIR, 'residual_chart_data.json'), 'w') as f:
    json.dump(residual_chart_data, f)
print("- Đã lưu dữ liệu biểu đồ sai số (residuals).")

print("\n===== QUÁ TRÌNH HUẤN LUYỆN VÀ XỬ LÝ HOÀN TẤT =====")
print(f"Tất cả file đã được tạo trong thư mục '{MODEL_DIR}' và '{DATA_DIR}'.")
print("Bây giờ bạn có thể chạy 'python app.py'")