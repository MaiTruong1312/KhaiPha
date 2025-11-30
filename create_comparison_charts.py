import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Cấu hình giao diện biểu đồ
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Đường dẫn đến thư mục chứa file JSON (tương ứng với cấu trúc folder của bạn)
MODEL_DIR = "models"
STATIC_DIR = "static"

# Đảm bảo thư mục static tồn tại
os.makedirs(STATIC_DIR, exist_ok=True)

def load_data():
    """Tải dữ liệu từ các file JSON đã được xuất ra bởi train_models.py"""
    try:
        # Tải dữ liệu biểu đồ test
        with open(os.path.join(MODEL_DIR, 'test_chart_data.json'), 'r') as f:
            test_data = json.load(f)
            
        # Tải dữ liệu so sánh metrics
        with open(os.path.join(MODEL_DIR, 'model_comparison.json'), 'r') as f:
            metrics_data = json.load(f)
            
        # Tải dữ liệu sai số (residuals)
        with open(os.path.join(MODEL_DIR, 'residual_chart_data.json'), 'r') as f:
            residual_data = json.load(f)
            
        return test_data, metrics_data, residual_data
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file JSON. Vui lòng chạy 'train_models.py' trước để tạo dữ liệu. Lỗi: {e}")
        return None, None, None

def plot_prediction_comparison(test_data):
    """Vẽ biểu đồ đường so sánh giá thực tế và dự đoán của cả 2 thuật toán"""
    dates = pd.to_datetime(test_data['dates'])
    actual = test_data['actual']
    pred_rf = test_data['predictions']['Random Forest']
    pred_xgb = test_data['predictions']['XGBoost']

    plt.figure(figsize=(14, 7))
    plt.plot(dates, actual, label='Thực tế (Actual)', color='black', alpha=0.6, linewidth=2)
    plt.plot(dates, pred_rf, label='Random Forest', color='blue', linestyle='--', alpha=0.8)
    plt.plot(dates, pred_xgb, label='XGBoost', color='red', linestyle='-.', alpha=0.8)

    plt.title('So sánh Dự báo: Thực tế vs Random Forest vs XGBoost', fontsize=16)
    plt.xlabel('Thời gian')
    plt.ylabel('Giá Vàng (USD/oz)')
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(STATIC_DIR, 'comparison_prediction_line.png')
    plt.savefig(save_path)
    print(f"Đã lưu biểu đồ: {save_path}")
    plt.close()

def plot_residuals_distribution(residual_data):
    """Vẽ biểu đồ phân phối sai số (Residuals Distribution)"""
    res_rf = residual_data['residuals']['Random Forest']
    res_xgb = residual_data['residuals']['XGBoost']

    plt.figure(figsize=(12, 6))
    sns.kdeplot(res_rf, fill=True, label='Random Forest Residuals', color='blue', alpha=0.3)
    sns.kdeplot(res_xgb, fill=True, label='XGBoost Residuals', color='red', alpha=0.3)

    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.title('Phân phối Sai số (Residuals Distribution)', fontsize=16)
    plt.xlabel('Sai số (Thực tế - Dự đoán)')
    plt.ylabel('Mật độ')
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(STATIC_DIR, 'comparison_residuals_dist.png')
    plt.savefig(save_path)
    print(f"Đã lưu biểu đồ: {save_path}")
    plt.close()

def plot_scatter_correlation(test_data):
    """Vẽ biểu đồ tương quan (Scatter Plot) giữa Thực tế và Dự đoán"""
    actual = test_data['actual']
    pred_rf = test_data['predictions']['Random Forest']
    pred_xgb = test_data['predictions']['XGBoost']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle('Biểu đồ Tương quan: Giá Thực tế vs. Giá Dự đoán', fontsize=18)

    # Biểu đồ cho Random Forest
    sns.scatterplot(x=actual, y=pred_rf, ax=axes[0], color='blue', alpha=0.5)
    min_val = min(min(actual), min(pred_rf))
    max_val = max(max(actual), max(pred_rf))
    axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    axes[0].set_title('Random Forest')
    axes[0].set_xlabel('Giá Thực tế')
    axes[0].set_ylabel('Giá Dự đoán')
    axes[0].set_aspect('equal', adjustable='box')


    # Biểu đồ cho XGBoost
    sns.scatterplot(x=actual, y=pred_xgb, ax=axes[1], color='red', alpha=0.5)
    min_val = min(min(actual), min(pred_xgb))
    max_val = max(max(actual), max(pred_xgb))
    axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    axes[1].set_title('XGBoost')
    axes[1].set_xlabel('Giá Thực tế')
    axes[1].set_ylabel('') # Tắt label y
    axes[1].set_aspect('equal', adjustable='box')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(STATIC_DIR, 'comparison_scatter_correlation.png')
    plt.savefig(save_path)
    print(f"Đã lưu biểu đồ: {save_path}")
    plt.close()

def plot_metrics_comparison(metrics_data):
    """So sánh chỉ số RMSE và R-Squared"""
    models = ['Random Forest', 'XGBoost']
    rmse = [metrics_data['Random Forest']['RMSE'], metrics_data['XGBoost']['RMSE']]
    r2 = [metrics_data['Random Forest']['R-squared'], metrics_data['XGBoost']['R-squared']]

    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Vẽ cột RMSE (Trục trái)
    rects1 = ax1.bar(x - width/2, rmse, width, label='RMSE (Thấp hơn là tốt)', color='salmon')
    ax1.set_ylabel('RMSE', color='salmon', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='salmon')
    ax1.set_title('So sánh Hiệu năng Mô hình (RMSE & R²)', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.bar_label(rects1, padding=3, fmt='%.2f')

    # Vẽ cột R2 (Trục phải)
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, r2, width, label='R² (Cao hơn là tốt)', color='skyblue')
    ax2.set_ylabel('R-Squared', color='skyblue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='skyblue')
    ax2.set_ylim(bottom=min(r2) - 0.001, top=max(r2) + 0.001)
    ax2.bar_label(rects2, padding=3, fmt='%.4f')
    
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    plt.tight_layout()
    save_path = os.path.join(STATIC_DIR, 'comparison_metrics_bar.png')
    plt.savefig(save_path)
    print(f"Đã lưu biểu đồ: {save_path}")
    plt.close()

if __name__ == "__main__":
    print("Đang tải dữ liệu...")
    test_data, metrics_data, residual_data = load_data()
    
    if test_data and metrics_data and residual_data:
        print("Đang vẽ biểu đồ so sánh đường (Line Chart)...")
        plot_prediction_comparison(test_data)
        
        print("Đang vẽ biểu đồ phân phối sai số (Residuals KDE)...")
        plot_residuals_distribution(residual_data)
        
        print("Đang vẽ biểu đồ tương quan (Scatter Plot)...")
        plot_scatter_correlation(test_data)
        
        print("Đang vẽ biểu đồ so sánh chỉ số (Bar Chart)...")
        plot_metrics_comparison(metrics_data)
        
        print("\nHoàn tất! Các file ảnh đã được lưu vào thư mục 'static'.")
    else:
        print("\nKết thúc do không tải được dữ liệu.")
