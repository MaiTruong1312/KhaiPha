from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.utils
from datetime import datetime, timedelta
import calendar
import os
import shutil
import tempfile # Thêm thư viện tempfile

app = Flask(__name__)

# --- HẰNG SỐ VÀ ĐƯỜNG DẪN (ĐÃ SỬA LỖI) ---
# SỬA LỖI: Tạo đường dẫn tuyệt đối để đảm bảo Flask luôn tìm thấy file
# Bất kể bạn chạy lệnh `python app.py` từ thư mục nào.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_FILE_PROCESSED = os.path.join(BASE_DIR, "data/preprocessed_gold_data.csv")

DEFAULT_TY_GIA_USD_VND = 26358
DEFAULT_SO_CHI_TREN_1_OZ = 8.29426

# ==============================================================================
# TẢI CÁC HIỆN VẬT (ARTIFACTS) KHI KHỞI ĐỘNG ỨNG DỤNG
# ==============================================================================
print("Đang tải models và artifacts...")
try:
    # Các đường dẫn này giờ đã là đường dẫn tuyệt đối, an toàn
    model_rf = joblib.load(os.path.join(MODEL_DIR, 'model_rf.pkl'))
    model_xgb = joblib.load(os.path.join(MODEL_DIR, 'model_xgb.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    
    with open(os.path.join(MODEL_DIR, 'model_comparison.json')) as f:
        model_comparison = json.load(f)
    
    with open(os.path.join(MODEL_DIR, 'latest_data.json')) as f:
        latest_data = json.load(f)
        # Chuyển đổi lại thành datetime object để tính toán
        latest_data['date_obj'] = datetime.strptime(latest_data['date'], '%Y-%m-%d')

    with open(os.path.join(MODEL_DIR, 'test_chart_data.json')) as f:
        test_chart_data = json.load(f)
        
    with open(os.path.join(MODEL_DIR, 'residual_chart_data.json')) as f:
        residual_chart_data = json.load(f)

    # Tải dữ liệu đã xử lý cho biểu đồ lịch sử
    df_processed = pd.read_csv(DATA_FILE_PROCESSED, parse_dates=['DATE'], index_col='DATE')
    
    # Lấy tên các cột feature (cần cho biểu đồ feature importance)
    features_columns = pd.read_csv(DATA_FILE_PROCESSED, nrows=0).columns.drop(['DATE', 'GOLD_PRICE']).tolist()

    best_model = model_rf if model_comparison['best_model'] == 'Random Forest' else model_xgb
    models = {'Random Forest': model_rf, 'XGBoost': model_xgb}
    best_model_name = model_comparison['best_model']
    
    print("✅ Models và data đã được tải thành công.")
except Exception as e:
    print(f"⚠️ LỖI NGHIÊM TRỌNG: Không thể tải models hoặc data.")
    print(f"   Lỗi chi tiết: {e}")
    print(f"   Vui lòng kiểm tra lại đường dẫn: {MODEL_DIR} và {DATA_FILE_PROCESSED}")
    # Trong môi trường production, bạn nên dừng ứng dụng ở đây
    
# ==============================================================================
# CÁC HÀM HỖ TRỢ (Lấy từ notebook)
# ==============================================================================
def convert_to_vnd_chi(gia_usd_oz, ty_gia, so_chi):
    gia_vnd_oz = gia_usd_oz * ty_gia
    gia_vnd_chi = gia_vnd_oz / so_chi
    return gia_vnd_chi

def simulate_future_price(days_diff):
    """Mô phỏng giá trong tương lai dựa trên xu hướng 30 ngày cuối."""
    # Logic mô phỏng đơn giản, chỉ dựa vào xu hướng
    random_factor = (np.random.rand() - 0.45) * 0.5 # Yếu tố ngẫu nhiên nhỏ
    trend_factor = latest_data['trend_30_day'] * days_diff
    
    # Đảm bảo giá không bao giờ âm
    predicted_price_usd_oz = max(0, latest_data['price'] + trend_factor + (random_factor * days_diff * 0.1))
    
    return predicted_price_usd_oz

# ==============================================================================
# CÁC HÀM TẠO BIỂU ĐỒ PLOTLY
# ==============================================================================
def create_chart(chart_type, selected_model_name=None):
    """Tạo các biểu đồ Plotly và chuyển thành JSON để gửi cho HTML."""
    
    # Cấu hình layout chung cho biểu đồ
    base_layout = {
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': '#333'},
        'xaxis': {'gridcolor': '#eee'},
        'yaxis': {'gridcolor': '#eee'},
        'margin': {'l': 50, 'r': 20, 't': 50, 'b': 40} # Tăng lề trái cho label
    }

    # Nếu không có model nào được chọn, mặc định là model tốt nhất
    if selected_model_name is None:
        selected_model_name = best_model_name

    if chart_type == 'history':
        fig = go.Figure(go.Scatter(
            x=df_processed.index,
            y=df_processed['GOLD_PRICE'],
            name='Giá Vàng Lịch Sử',
            line=dict(color='#D97706') # Màu vàng đậm (Tailwind amber-600)
        ))
        fig.update_layout(
            title='Biểu Đồ Lịch Sử Giá Vàng (Đã xử lý)',
            xaxis_title='Ngày',
            yaxis_title='Giá (USD/oz)',
            **base_layout
        )
    
    elif chart_type == 'prediction_compare':
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=test_chart_data['dates'], y=test_chart_data['actual'], name='Giá Thực Tế (Test)',
            line=dict(color='#2563EB', width=2) # Màu xanh (Tailwind blue-600)
        ))
        fig.add_trace(go.Scatter(
            x=test_chart_data['dates'], y=test_chart_data['predictions'][selected_model_name], name=f'Dự Đoán ({selected_model_name})',
            line=dict(color='#DC2626', width=2, dash='dash') # Màu đỏ (Tailwind red-600)
        ))
        fig.update_layout(
            title=f'So Sánh: Giá Thực Tế vs. Dự Đoán ({selected_model_name})',
            xaxis_title='Ngày',
            yaxis_title='Giá (USD/oz)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            **base_layout
        )
    
    elif chart_type == 'residuals':
        fig = go.Figure(go.Scatter(
            x=residual_chart_data['dates'],
            y=residual_chart_data['residuals'][selected_model_name],
            mode='markers',
            name='Sai số',
            marker=dict(color='#EF4444', opacity=0.6) # Màu đỏ (Tailwind red-500)
        ))
        fig.add_hline(y=0, line_width=2, line_dash="dash", line_color="grey")
        fig.update_layout(
            title=f'Phân Tích Sai Số Dự Đoán (Residuals) - {selected_model_name}',
            xaxis_title='Ngày',
            yaxis_title='Sai số (Thực tế - Dự đoán)',
            showlegend=False,
            **base_layout
        )


    elif chart_type in ['fi_rf', 'fi_xgb']:
        model_name = 'Random Forest' if chart_type == 'fi_rf' else 'XGBoost'
        model = models[model_name]
        
        # Đảm bảo model có feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            df_importance = pd.DataFrame({'feature': features_columns, 'importance': importances})
            df_importance = df_importance.sort_values(by='importance', ascending=True) # Sắp xếp tăng dần cho bar chart
            
            fig = go.Figure(go.Bar(
                x=df_importance['importance'], 
                y=df_importance['feature'], 
                orientation='h', # Biểu đồ thanh ngang
                marker_color='#10B981' # Màu xanh lá (Tailwind emerald-500)
            ))
            fig.update_layout(
                title=f"Tầm quan trọng của Yếu tố - {model_name}",
                xaxis_title='Tầm quan trọng',
                yaxis_title='Yếu tố',
                **base_layout
            )
        else:
             fig = go.Figure().update_layout(title=f"Không thể lấy Feature Importance cho {model_name}", **base_layout)
    
    else:
        fig = go.Figure() # Trả về biểu đồ rỗng nếu lỗi

    # Chuyển biểu đồ thành JSON để JavaScript (plotly.js) có thể đọc
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# ==============================================================================
# CÁC ROUTE (TRANG WEB) CỦA FLASK
# ==============================================================================

@app.route('/', methods=['GET'])
def index():
    """Trang chủ, hiển thị tất cả biểu đồ và form."""

    # Lấy model được chọn từ query parameter, nếu không có thì dùng model tốt nhất
    selected_model = request.args.get('model', default=best_model_name, type=str)
    
    # Tạo JSON cho các biểu đồ
    chart_hist_json = create_chart('history')
    chart_pred_json = create_chart('prediction_compare', selected_model)
    chart_residuals_json = create_chart('residuals', selected_model)
    chart_fi_rf_json = create_chart('fi_rf')
    chart_fi_xgb_json = create_chart('fi_xgb')
    
    # Các giá trị mặc định cho form
    default_values = {
        'ty_gia': DEFAULT_TY_GIA_USD_VND,
        'so_chi': DEFAULT_SO_CHI_TREN_1_OZ,
        'date_input': (latest_data['date_obj'] + timedelta(days=1)).strftime('%Y-%m-%d'),
        'month_input': (latest_data['date_obj'] + timedelta(days=1)).strftime('%Y-%m')
    }

    return render_template('index.html',
                           model_comparison=model_comparison,
                           chart_hist_json=chart_hist_json,
                           selected_model=selected_model, # Truyền model đã chọn ra view
                           chart_pred_json=chart_pred_json,
                           chart_residuals_json=chart_residuals_json,
                           chart_fi_rf_json=chart_fi_rf_json,
                           chart_fi_xgb_json=chart_fi_xgb_json,
                           defaults=default_values,
                           active_tab='tab-1' # Mặc định mở tab 1
                           )

@app.route('/predict_day', methods=['POST'])
def predict_day():
    """Xử lý form dự đoán theo ngày."""
    # Lấy model được chọn từ form ẩn, nếu không có thì dùng model tốt nhất
    selected_model = request.form.get('selected_model', default=best_model_name, type=str)

    prediction_result_day = None
    error_day = None
    
    # Lấy dữ liệu từ form
    form_values = request.form.to_dict()
    
    try:
        selected_date_str = request.form['date_input']
        ty_gia = float(request.form['ty_gia'])
        so_chi = float(request.form['so_chi'])
        
        selected_date_dt = datetime.strptime(selected_date_str, '%Y-%m-%d')
        
        if selected_date_dt <= latest_data['date_obj']:
            raise ValueError(f"Vui lòng chọn một ngày trong tương lai (sau ngày {latest_data['date']}).")

        time_diff = selected_date_dt - latest_data['date_obj']
        days_diff = time_diff.days
        
        predicted_price_usd_oz = simulate_future_price(days_diff)
        predicted_price_vnd_chi = convert_to_vnd_chi(predicted_price_usd_oz, ty_gia, so_chi)

        prediction_result_day = {
            'date': selected_date_dt.strftime('%d-%m-%Y'),
            'usd_oz': f"{predicted_price_usd_oz:.2f}",
            'vnd_chi': f"{predicted_price_vnd_chi:,.0f}",
            'ty_gia': f"{ty_gia:,.0f}"
        }

    except Exception as e:
        error_day = str(e)
    
    # Render lại toàn bộ trang với kết quả dự đoán
    return render_template('index.html',
                           model_comparison=model_comparison,
                           features_list=features_columns,
                           selected_model=selected_model,
                           chart_hist_json=create_chart('history', selected_model),
                           chart_residuals_json=create_chart('residuals', selected_model),
                           chart_pred_json=create_chart('prediction_compare', selected_model),
                           chart_fi_rf_json=create_chart('fi_rf'),
                           chart_fi_xgb_json=create_chart('fi_xgb'),
                           defaults=form_values, # Gửi lại giá trị người dùng đã nhập
                           prediction_result_day=prediction_result_day,
                           error_day=error_day,
                           active_tab='tab-2' # Báo cho HTML biết tab nào đang active
                           )

@app.route('/predict_month', methods=['POST'])
def predict_month():
    """Xử lý form dự đoán theo tháng."""
    # Lấy model được chọn từ form ẩn, nếu không có thì dùng model tốt nhất
    selected_model = request.form.get('selected_model', default=best_model_name, type=str)

    prediction_result_month = None
    error_month = None

    # Lấy dữ liệu từ form
    form_values = request.form.to_dict()
    
    try:
        year_month_str = request.form['month_input']
        ty_gia = float(request.form['ty_gia'])
        so_chi = float(request.form['so_chi'])
        
        year, month = map(int, year_month_str.split('-'))
        _, num_days_in_month = calendar.monthrange(year, month)
        
        monthly_predictions_usd = []
        for day in range(1, num_days_in_month + 1):
            current_date_dt = datetime(year, month, day)
            if current_date_dt <= latest_data['date_obj']:
                continue # Bỏ qua các ngày trong quá khứ
            
            time_diff = current_date_dt - latest_data['date_obj']
            days_diff = time_diff.days
            monthly_predictions_usd.append(simulate_future_price(days_diff))
        
        if not monthly_predictions_usd:
             raise ValueError(f"Tháng {year_month_str} đã hoàn toàn ở trong quá khứ hoặc không hợp lệ.")

        avg_price_usd_oz = np.mean(monthly_predictions_usd)
        avg_price_vnd_chi = convert_to_vnd_chi(avg_price_usd_oz, ty_gia, so_chi)

        prediction_result_month = {
            'month': year_month_str,
            'usd_oz': f"{avg_price_usd_oz:.2f}",
            'vnd_chi': f"{avg_price_vnd_chi:,.0f}",
            'ty_gia': f"{ty_gia:,.0f}"
        }

    except Exception as e:
        error_month = str(e)

    # Render lại toàn bộ trang với kết quả dự đoán
    return render_template('index.html',
                           model_comparison=model_comparison,
                           features_list=features_columns,
                           selected_model=selected_model,
                           chart_hist_json=create_chart('history', selected_model),
                           chart_residuals_json=create_chart('residuals', selected_model),
                           chart_pred_json=create_chart('prediction_compare', selected_model),
                           chart_fi_rf_json=create_chart('fi_rf'),
                           chart_fi_xgb_json=create_chart('fi_xgb'),
                           defaults=form_values, # Gửi lại giá trị người dùng đã nhập
                           prediction_result_month=prediction_result_month,
                           error_month=error_month,
                           active_tab='tab-3' # Báo cho HTML biết tab nào đang active
                           )

@app.route('/methodology')
def methodology():
    """Trang giải thích phương pháp luận."""
    return render_template('methodology.html')

@app.route('/download')
def download_project():
    """Gửi file zip đã được nén sẵn cho người dùng."""
    try:
        # Gửi file 'gold-price-prediction-project.zip' từ thư mục 'static'
        return send_from_directory(directory=STATIC_DIR, path='gold-price-prediction-project.zip', as_attachment=True)
    except FileNotFoundError:
        return "Lỗi: Không tìm thấy file 'gold-price-prediction-project.zip' trong thư mục 'static'. Vui lòng tạo file nén trước.", 404
    except Exception as e:
        return str(e)

# --- Chạy ứng dụng ---
if __name__ == '__main__':
    # Bật debug=True chỉ khi phát triển. 
    # Bật use_reloader=False để tránh việc tải model 2 lần khi khởi động.
    app.run(debug=True, use_reloader=False)