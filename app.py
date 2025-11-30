from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.utils
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import calendar
import os
import shutil
 
app = Flask(__name__)
 

 
# --- CÁC HẰNG SỐ VÀ CẤU HÌNH ---
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
 
MODEL_DIR = os.path.join(BASE_DIR, "models")
 
STATIC_DIR = os.path.join(BASE_DIR, "static")
  
DEFAULT_TY_GIA_USD_VND = 26358
 
DEFAULT_SO_CHI_TREN_1_OZ = 8.29426
FILE_GOLD_SILVER_RAW = os.path.join(BASE_DIR, 'data/Gold-Silver-GeopoliticalRisk_HistoricalData.csv')
FILE_GPRD_FEATURES_RAW = os.path.join(BASE_DIR, 'data/df_recent_preprocessed.csv')

def load_raw_data():
    """Tải dữ liệu thô từ các file CSV."""
    print(f"Đang tải dữ liệu giá thô từ: {FILE_GOLD_SILVER_RAW}...")
    try:
        df_price_cols = pd.read_csv(FILE_GOLD_SILVER_RAW, nrows=0).columns.tolist()
        price_use_cols = ['DATE', 'GOLD_PRICE']
        if 'SILVER_PRICE' in df_price_cols:
            print("- Tìm thấy cột 'SILVER_PRICE'.")
            price_use_cols.append('SILVER_PRICE')
        else:
            print("⚠️ CẢNH BÁO: Không tìm thấy cột 'SILVER_PRICE' trong file.")

        df_price = pd.read_csv(FILE_GOLD_SILVER_RAW, usecols=price_use_cols, parse_dates=['DATE']).sort_values(by='DATE', ascending=True)
    except Exception as e:
        print(f"❌ LỖI: Không thể đọc file giá thô. Lỗi: {e}")
        return None, None

    df_gprd = None
    if os.path.exists(FILE_GPRD_FEATURES_RAW):
        print(f"Đang tải dữ liệu GPRD thô từ: {FILE_GPRD_FEATURES_RAW}...")
        try:
            if 'GPRD' in pd.read_csv(FILE_GPRD_FEATURES_RAW, nrows=0).columns:
                df_gprd = pd.read_csv(FILE_GPRD_FEATURES_RAW, usecols=['date', 'GPRD'], parse_dates=['date']).sort_values(by='date', ascending=True)
                print("- Tìm thấy cột 'GPRD'.")
        except Exception as e:
            print(f"⚠️ CẢNH BÁO: Không thể đọc file GPRD. Lỗi: {e}.")
    else:
        print(f"ℹ️ INFO: File '{FILE_GPRD_FEATURES_RAW}' không tồn tại, bỏ qua dữ liệu GPRD.")
    return df_price, df_gprd

def process_in_memory_data(df_price, df_gprd):
    """Thực hiện merge, feature engineering và clean cho dữ liệu trong bộ nhớ."""
    if df_price is None: return None
    
    # 1. Merge Data
    print("Đang gộp dữ liệu...")
    df_merged = df_price
    if df_gprd is not None:
        df_merged = pd.merge(df_price, df_gprd, left_on='DATE', right_on='date', how='left').drop(columns=['date'])

    # 2. Feature Engineering cho biểu đồ
    print("Đang tạo các đặc trưng cho biểu đồ...")
    df_featured = df_merged.copy()
    if 'SILVER_PRICE' in df_featured.columns:
        df_featured['GS_RATIO'] = df_featured['GOLD_PRICE'] / (df_featured['SILVER_PRICE'] + 1e-6)
    
    # 3. Clean and Set Index
    print("Đang hoàn thiện dữ liệu...")
    df_final = df_featured.set_index('DATE')
    # Fill NA cho mục đích hiển thị
    df_final = df_final.fillna(method='ffill').fillna(method='bfill')
    return df_final

# ==============================================================================
# TẢI CÁC HIỆN VẬT (ARTIFACTS) KHI KHỞI ĐỘNG ỨNG DỤNG
# ==============================================================================
print("===== BẮT ĐẦU QUÁ TRÌNH TIỀN XỬ LÝ DỮ LIỆU (IN-APP) =====")
df_price_raw, df_gprd_raw = load_raw_data()
df_processed = process_in_memory_data(df_price_raw, df_gprd_raw)

if df_processed is not None:
    print(f"✅ TIỀN XỬ LÝ (IN-APP) HOÀN TẤT. Các cột có sẵn: {df_processed.columns.tolist()}")
else:
    print("❌ LỖI: Không thể xử lý dữ liệu trong ứng dụng.")
    df_processed = pd.DataFrame(columns=['DATE','GOLD_PRICE', 'SILVER_PRICE', 'GS_RATIO', 'GPRD']).set_index('DATE')

print("\nĐang tải models và artifacts...")
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

    # Lấy tên các cột feature mà model đã được huấn luyện
    # QUAN TRỌNG: Danh sách này phải khớp với các feature được sử dụng trong train_models.py
    model_feature_columns = [
        'GOLD_PRICE_lag1', 'GOLD_PRICE_lag7', 'GOLD_PRICE_lag30',
        'GOLD_PRICE_roll_mean_7', 'GOLD_PRICE_roll_mean_30',
        'GOLD_PRICE_roll_std_7', 'GOLD_PRICE_roll_std_30',
        'month', 'day_of_week', 'day_of_year'
    ]

    best_model = model_rf if model_comparison['best_model'] == 'Random Forest' else model_xgb
    models = {'Random Forest': model_rf, 'XGBoost': model_xgb}
    best_model_name = model_comparison['best_model']
    
    print("✅ Models và data đã được tải thành công.")
except Exception as e:
    print(f"⚠️ LỖI NGHIÊM TRỌNG: Không thể tải models hoặc data.")
    print(f"   Lỗi chi tiết: {e}")
    print(f"   Vui lòng kiểm tra lại đường dẫn: {MODEL_DIR}")
    
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
        'margin': {'l': 50, 'r': 20, 't': 50, 'b': 40} 
    }

    # Nếu không có model nào được chọn, mặc định là model tốt nhất
    if selected_model_name is None:
        selected_model_name = best_model_name

    if chart_type == 'history':
        fig = go.Figure(go.Scatter(
            x=df_processed.index,
            y=df_processed['GOLD_PRICE'],
            name='Giá Vàng Lịch Sử',
            line=dict(color='#D97706')
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
            line=dict(color='#2563EB', width=2) 
        ))
        fig.add_trace(go.Scatter(
            x=test_chart_data['dates'], y=test_chart_data['predictions'][selected_model_name], name=f'Dự Đoán ({selected_model_name})',
            line=dict(color='#DC2626', width=2, dash='dash') 
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
            marker=dict(color='#EF4444', opacity=0.6) 
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
            
            df_importance = pd.DataFrame({'feature': model_feature_columns, 'importance': importances})
            df_importance = df_importance.sort_values(by='importance', ascending=True) 
            
            fig = go.Figure(go.Bar(
                x=df_importance['importance'], 
                y=df_importance['feature'], 
                orientation='h', 
                marker_color='#10B981' 
            ))
            fig.update_layout(
                title=f"Tầm quan trọng của Yếu tố - {model_name}",
                xaxis_title='Tầm quan trọng',
                yaxis_title='Yếu tố',
                **base_layout
            )
        else:
             fig = go.Figure().update_layout(title=f"Không thể lấy Feature Importance cho {model_name}", **base_layout)

    elif chart_type == 'scatter_correlation':
        fig = go.Figure()
        actual = test_chart_data['actual']
        prediction = test_chart_data['predictions'][selected_model_name]

        # Thêm điểm scatter
        fig.add_trace(go.Scatter(
            x=actual, y=prediction, mode='markers', name='Dự đoán',
            marker=dict(color='#DC2626', opacity=0.5)
        ))

        # Thêm đường y=x
        min_val = min(min(actual), min(prediction))
        max_val = max(max(actual), max(prediction))
        fig.add_shape(type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color='grey', dash='dash'))

        fig.update_layout(
            title=f'Tương quan Thực tế vs. Dự đoán - {selected_model_name}',
            xaxis_title='Giá Thực tế (USD/oz)',
            yaxis_title='Giá Dự đoán (USD/oz)',
            **base_layout
        )
    
    elif chart_type == 'metrics_comparison':
        metrics = model_comparison
        model_names = ['Random Forest', 'XGBoost']
        rmse = [metrics['Random Forest']['RMSE'], metrics['XGBoost']['RMSE']]
        r2 = [metrics['Random Forest']['R-squared'], metrics['XGBoost']['R-squared']]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add RMSE bars
        fig.add_trace(go.Bar(
            x=model_names, y=rmse, name='RMSE (Thấp hơn là tốt)',
            marker_color='salmon', text=[f'{x:.2f}' for x in rmse], textposition='auto'
        ), secondary_y=False)

        # Add R-squared bars
        fig.add_trace(go.Bar(
            x=model_names, y=r2, name='R² (Cao hơn là tốt)',
            marker_color='skyblue', text=[f'{x:.4f}' for x in r2], textposition='auto'
        ), secondary_y=True)

        fig.update_layout(
            title_text='So sánh Hiệu năng Mô hình (RMSE & R²)',
            barmode='group',
            **base_layout
        )
        # Set y-axes titles
        fig.update_yaxes(title_text="RMSE", secondary_y=False)
        fig.update_yaxes(title_text="R-Squared", secondary_y=True)

    else:
        fig = go.Figure() 

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
                           features_list=model_feature_columns,
                           chart_hist_json=chart_hist_json,
                           selected_model=selected_model, 
                           chart_pred_json=chart_pred_json,
                           chart_residuals_json=chart_residuals_json,
                           chart_fi_rf_json=chart_fi_rf_json,
                           chart_fi_xgb_json=chart_fi_xgb_json,
                           defaults=default_values,
                           active_tab='tab-1' 
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
                           features_list=model_feature_columns,
                           selected_model=selected_model,
                           chart_hist_json=create_chart('history', selected_model),
                           chart_residuals_json=create_chart('residuals', selected_model),
                           chart_pred_json=create_chart('prediction_compare', selected_model),
                           chart_fi_rf_json=create_chart('fi_rf'),
                           chart_fi_xgb_json=create_chart('fi_xgb'),
                           defaults=form_values, 
                           prediction_result_day=prediction_result_day,
                           error_day=error_day,
                           active_tab='tab-2' 
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
                continue 
            
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
                           features_list=model_feature_columns,
                           selected_model=selected_model,
                           chart_hist_json=create_chart('history', selected_model),
                           chart_residuals_json=create_chart('residuals', selected_model),
                           chart_pred_json=create_chart('prediction_compare', selected_model),
                           chart_fi_rf_json=create_chart('fi_rf'),
                           chart_fi_xgb_json=create_chart('fi_xgb'),
                           defaults=form_values, 
                           prediction_result_month=prediction_result_month,
                           error_month=error_month,
                           active_tab='tab-3' 
                           )

@app.route('/methodology')
def methodology():
    """Trang giải thích phương pháp luận."""
    return render_template('methodology.html')

@app.route('/comparison')
def comparison():
    """Trang so sánh chi tiết các thuật toán."""
    return render_template('comparison.html', model_comparison=model_comparison)

@app.route('/visual_comparison')
def visual_comparison():
    """Trang hiển thị các biểu đồ so sánh trực quan."""
    # Lấy model được chọn từ query parameter, nếu không có thì dùng model tốt nhất
    selected_model = request.args.get('model', default=best_model_name, type=str)

    # Tạo JSON cho các biểu đồ
    chart_pred_json = create_chart('prediction_compare', selected_model)
    chart_residuals_json = create_chart('residuals', selected_model)
    chart_scatter_json = create_chart('scatter_correlation', selected_model)
    chart_metrics_json = create_chart('metrics_comparison')

    return render_template('visual_comparison.html',
                           chart_pred_json=chart_pred_json,
                           chart_residuals_json=chart_residuals_json,
                           chart_scatter_json=chart_scatter_json,
                           chart_metrics_json=chart_metrics_json,
                           selected_model=selected_model,
                           model_names=['Random Forest', 'XGBoost']
                           )

@app.route('/history')
def history():
    """Trang phân tích lịch sử chuyên sâu."""
    shown_series = request.args.getlist('series')
    shown_mas = request.args.getlist('ma')
    normalize = request.args.get('normalize', 'false').lower() == 'true'

    if not shown_series:
        shown_series = ['gold']

    fig = go.Figure()
    
    series_map = {
        'gold': ('Giá Vàng (USD/oz)', 'GOLD_PRICE', '#facc15'),
        'silver': ('Giá Bạc (USD/oz)', 'SILVER_PRICE', '#9ca3af'),
        'ratio': ('Tỷ lệ Vàng/Bạc', 'GS_RATIO', '#a78bfa'),
        'gprd': ('Chỉ số GPRD', 'GPRD', '#fb923c')
    }

    df_plot = df_processed.copy()
    
    if normalize:
        main_yaxis_title = "Thay đổi theo % (từ ngày bắt đầu)"
        for series_code in shown_series:
            col_name = series_map.get(series_code, [None, None])[1]
            if col_name and col_name in df_plot.columns:
                first_valid_value = df_plot[col_name].dropna().iloc[0]
                if first_valid_value != 0:
                    df_plot[col_name] = (df_plot[col_name] / first_valid_value) * 100
                else:
                    df_plot[col_name] = 0 
    else:
        main_yaxis_title = "Giá trị"

    for i, series_code in enumerate(shown_series):
        display_name, col_name, color = series_map.get(series_code, (None, None, None))
        if display_name and col_name in df_plot.columns:
            fig.add_trace(go.Scatter(
                x=df_plot.index,
                y=df_plot[col_name],
                name=display_name,
                line=dict(color=color, width=2),
                yaxis=f'y{i+1}' if not normalize and len(shown_series) > 1 else 'y1'
            ))

    ma_colors = ['#3b82f6', '#16a34a', '#ef4444']
    if 'gold' in shown_series:
        for i, ma_period_str in enumerate(shown_mas):
            try:
                ma_period = int(ma_period_str)
                ma_series = df_processed['GOLD_PRICE'].rolling(window=ma_period).mean()
                
                if normalize:
                    # SỬA LỖI: Chuẩn hóa đường MA với giá trị hợp lệ đầu tiên CỦA CHÍNH NÓ
                    first_ma_value = ma_series.dropna().iloc[0]
                    if first_ma_value != 0:
                        ma_series = (ma_series / first_ma_value) * 100
                    else:
                        ma_series = 0
                
                fig.add_trace(go.Scatter(
                    x=df_plot.index,
                    y=ma_series,
                    name=f'MA {ma_period} (Vàng)',
                    line=dict(color=ma_colors[i % len(ma_colors)], width=1.5, dash='dash'),
                    yaxis='y1' 
                ))
            except (ValueError, IndexError):
                continue

    fig.update_layout(
        title_text='Phân Tích Lịch Sử Đa Yếu Tố',
        xaxis_title='Ngày',
        yaxis_title=main_yaxis_title,
        legend_title_text='Các chuỗi dữ liệu',
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(255,255,255,1)',
        font_color='#333',
        xaxis_gridcolor='#eee',
        yaxis_gridcolor='#eee',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    if not normalize and len(shown_series) > 1:
        layout_updates = {}
        num_right_axes = len(shown_series) - 1
        xaxis_domain_end = 1.0 - (num_right_axes * 0.05)
        layout_updates['xaxis'] = {'domain': [0, xaxis_domain_end]}

        for i, series_code in enumerate(shown_series):
            display_name, _, color = series_map.get(series_code, (None, None, '#333'))
            axis_name = f'yaxis{i+1}'
            
            if i == 0:  
                axis_config = {
                    'title': {'text': display_name, 'font': {'color': color}},
                    'tickfont': {'color': color},
                    'anchor': 'x',
                    'side': 'left'
                }
            else:  
                axis_config = {
                    'title': {'text': display_name, 'font': {'color': color}},
                    'tickfont': {'color': color},
                    'overlaying': 'y',
                    'side': 'right',
                    'showgrid': False,
                    'anchor': 'free',
                    'position': xaxis_domain_end + (0.05 * i) 
                }
            layout_updates[axis_name] = axis_config
        
        fig.update_layout(**layout_updates)


    chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('history.html', 
                           chart_json=chart_json,
                           selected_series=shown_series,
                           selected_mas=shown_mas,
                           is_normalized=normalize
                           )

@app.route('/download')
def download_project():
    """Gửi file zip đã được nén sẵn cho người dùng."""
    try:
        return send_from_directory(directory=STATIC_DIR, path='gold-price-prediction-project.zip', as_attachment=True)
    except FileNotFoundError:
        return "Lỗi: Không tìm thấy file 'gold-price-prediction-project.zip' trong thư mục 'static'. Vui lòng tạo file nén trước.", 404
    except Exception as e:
        return str(e)

# --- Chạy ứng dụng ---
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)