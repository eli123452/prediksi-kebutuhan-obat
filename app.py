import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from io import BytesIO
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

# FUNGSI BARU UNTUK VISUALISASI DAN EVALUASI
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
# Import metrik yang spesifik
from sklearn.metrics import accuracy_score, precision_score, recall_score
import plotly.figure_factory as ff

# =================================================================================================
# FUNGSI BANTUAN
# =================================================================================================

@st.cache_resource
def load_model():
    """Memuat model, label encoder, dan fitur yang telah di-train."""
    try:
        model = joblib.load('model.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        model_features = joblib.load('model_features.pkl')
        return model, label_encoder, model_features
    except FileNotFoundError:
        st.error("File model tidak ditemukan. Pastikan file 'model.pkl', 'label_encoder.pkl', dan 'model_features.pkl' ada.")
        return None, None, None

def to_excel(df):
    """Mengkonversi DataFrame ke format Excel dalam memory."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Hasil Prediksi')
    return output.getvalue()

def to_pdf(df, title):
    """
    Mengkonversi DataFrame ke format PDF dengan penyesuaian lebar kolom otomatis.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter),
                            rightMargin=0.5*inch, leftMargin=0.5*inch,
                            topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    styles = getSampleStyleSheet()
    elements = []

    # Judul Laporan
    title_style = styles['h1']
    elements.append(Paragraph(title, title_style))
    elements.append(Spacer(1, 0.2*inch))

    # Menghitung lebar yang tersedia untuk tabel
    page_width, page_height = landscape(letter)
    available_width = page_width - doc.leftMargin - doc.rightMargin

    # Menyesuaikan lebar setiap kolom secara proporsional
    num_cols = len(df.columns)
    col_widths = [available_width / num_cols] * num_cols

    # Membungkus teks di setiap sel agar bisa turun baris (text wrapping)
    # dan mengatur ukuran font lebih kecil
    cell_style = styles['Normal']
    cell_style.fontSize = 8
    cell_style.leading = 10

    # Mengubah isi dataframe menjadi objek Paragraph
    data_pdf = [[Paragraph(str(col), cell_style) for col in df.columns]]
    for _, row in df.iterrows():
        data_pdf.append([Paragraph(str(cell), cell_style) for cell in row])
        
    table = Table(data_pdf, colWidths=col_widths, hAlign='CENTER')

    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#FFFFFF")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F0F2F6')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ])
    table.setStyle(style)

    elements.append(table)
    doc.build(elements)
    
    return buffer.getvalue()


# =================================================================================================
# KONFIGURASI HALAMAN DAN CSS
# =================================================================================================

st.set_page_config(
    page_title="Prediksi Stok Obat - Apotek Barokah Farma",
    layout="wide"
)

st.markdown("""
<style>
    html, body, [class*="st-"] { font-family: 'IBM Plex Sans', sans-serif; }
    [data-testid="stSidebar"] { background-color: #4B0082; }
    [data-testid="stSidebar"] div, [data-testid="stSidebar"] p, [data-testid="stSidebar"] li, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] [data-testid="stHeading"] { color: #FFFFFF !important; }
    [data-testid="stSidebar"] .stAlert { background-color: rgba(255, 255, 255, 0.15); }
    .main-header { font-size: 38px; font-weight: bold; color: #4B0082; text-align: center; margin-bottom: 30px; }
    .card { background-color: white; border-radius: 10px; padding: 25px; margin-bottom: 20px; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1); border: 1px solid #E0E0E0; }
    .stButton>button { background-color: #5D3FD3; color: white; border-radius: 8px; border: none; padding: 12px 28px; font-weight: bold; width: 100%; transition: background-color 0.3s ease; }
    .stButton>button:hover { background-color: #4B0082; color: white; }
    div[data-testid*="stButton"] button[kind="secondary"] { background-color: #F0F2F6; color: #4B0082; border: 1px solid #5D3FD3; }
    div[data-testid*="stButton"] button[kind="secondary"]:hover { background-color: #E0E0E0; border: 1px solid #4B0082; color: #4B0082; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px; padding: 10px 15px; }
    .stTabs [aria-selected="true"] { background-color: #F0F2F6; color: #4B0082; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

model, le, model_features = load_model()

# =================================================================================================
# UI - SIDEBAR
# =================================================================================================
with st.sidebar:
    st.title("Apotek Barokah Farma")
    st.header("Sistem Prediksi Stok Obat")
    st.info("Aplikasi ini menggunakan model *Decision Tree* untuk memprediksi kebutuhan restok obat berdasarkan data historis penjualan.")
    st.header("Cara Penggunaan")
    st.markdown("""
    1.  **Pilih Tab:** 'Lakukan Prediksi' atau 'Evaluasi Model'.
    2.  **Untuk Prediksi:** Unggah file atau input manual data baru.
    3.  **Untuk Evaluasi:** Unggah file data historis yang sudah memiliki label asli.
    4.  Klik tombol prediksi atau evaluasi.
    5.  Hasil akan ditampilkan di halaman.
    """)

# =================================================================================================
# UI - HALAMAN UTAMA
# =================================================================================================
st.markdown('<p class="main-header"><h2><center>PREDIKSI KEBUTUHAN STOK OBAT<center/></h2></p>', unsafe_allow_html=True)

tab_prediksi, tab_evaluasi = st.tabs(["Lakukan Prediksi", "Evaluasi Model"])

# --- TAB 1: LAKUKAN PREDIKSI ---
with tab_prediksi:
    with st.container(border=False):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        input_method = st.radio("Pilih Metode Input Data:", ("Unggah File (CSV/Excel)", "Input Data Manual"), horizontal=True)
        st.write("---")
        
        data_to_predict = None

        if input_method == "Unggah File (CSV/Excel)":
            st.subheader("Unggah File Data Obat")
            uploaded_file = st.file_uploader("Pilih file .csv atau .xlsx", type=['csv', 'xlsx'], label_visibility="collapsed")
            if uploaded_file:
                try:
                    df_input = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                    st.success(f"File '{uploaded_file.name}' berhasil diunggah. Terdapat {len(df_input)} baris data.")
                    st.markdown("##### Pemetaan Kolom")
                    col1, col2 = st.columns(2)
                    available_cols = ['-'] + df_input.columns.tolist()
                    col_stok_awal = col1.selectbox("Pilih Kolom Stok Awal:", available_cols)
                    col_terjual = col2.selectbox("Pilih Kolom Terjual:", available_cols)
                    if col_stok_awal != '-' and col_terjual != '-':
                        if st.button("Lakukan Prediksi", key="predict_upload"):
                            identifier_cols = [col for col in df_input.columns if col not in [col_stok_awal, col_terjual]]
                            ordered_cols = identifier_cols + [col_stok_awal, col_terjual]
                            data_to_predict = df_input[ordered_cols].copy()
                            data_to_predict.rename(columns={col_stok_awal: 'STOK AWAL', col_terjual: 'TERJUAL'}, inplace=True)
                except Exception as e:
                    st.error(f"Gagal memproses file. Pastikan format file benar. Error: {e}")
        else:
            st.subheader("Input Data Obat Manual")
            if 'manual_rows' not in st.session_state:
                st.session_state.manual_rows = [{'id': 0}]
            if 'next_id' not in st.session_state:
                st.session_state.next_id = 1

            def remove_row(row_id):
                st.session_state.manual_rows = [row for row in st.session_state.manual_rows if row['id'] != row_id]
            for i, row in enumerate(st.session_state.manual_rows):
                row_id = row['id']
                col1, col2, col3, col4 = st.columns([2, 2, 3, 1])
                row['nama_item'] = col1.text_input("Nama Item", placeholder=f"Item #{i+1}", key=f"nama_{row_id}", label_visibility="collapsed")
                row['stok_awal'] = col2.number_input("Stok Awal", min_value=0, step=1, key=f"stok_{row_id}", label_visibility="collapsed")
                row['terjual'] = col3.number_input("Terjual", min_value=0, step=1, key=f"terjual_{row_id}", label_visibility="collapsed")
                col4.button("Hapus", key=f"del_{row_id}", on_click=remove_row, args=(row_id,), type="secondary")
            def add_row():
                st.session_state.manual_rows.append({'id': st.session_state.next_id})
                st.session_state.next_id += 1
            col_add, col_predict_manual = st.columns([1, 1])
            col_add.button("Tambah Item", on_click=add_row, type="secondary")
            if col_predict_manual.button("Lakukan Prediksi", key="predict_manual"):
                valid_rows = [row for row in st.session_state.manual_rows if row.get('stok_awal') is not None and row.get('terjual') is not None and row.get('nama_item')]
                if valid_rows:
                    df_manual = pd.DataFrame(valid_rows)
                    data_to_predict = df_manual.rename(columns={'stok_awal': 'STOK AWAL', 'terjual': 'TERJUAL', 'nama_item': 'NAMA ITEM'}).drop(columns=['id'])
                else:
                    st.warning("Mohon isi semua kolom pada setiap baris.")
        st.markdown('</div>', unsafe_allow_html=True)

    if data_to_predict is not None and not data_to_predict.empty and model is not None:
        try:
            df_predict = data_to_predict.copy()
            df_predict['STOK AWAL'] = pd.to_numeric(df_predict['STOK AWAL'], errors='coerce')
            df_predict['TERJUAL'] = pd.to_numeric(df_predict['TERJUAL'], errors='coerce')

            if df_predict['STOK AWAL'].isnull().any() or df_predict['TERJUAL'].isnull().any():
                st.error("Data 'Stok Awal' atau 'Terjual' mengandung nilai yang tidak valid (bukan angka).")
            else:
                df_predict['SISA STOK'] = df_predict['STOK AWAL'] - df_predict['TERJUAL']
                X_pred = df_predict[model_features]
                predictions = le.inverse_transform(model.predict(X_pred))
                df_predict['STATUS KEBUTUHAN'] = predictions
                st.success("Prediksi berhasil dilakukan!")
                
                with st.container(border=False):
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Visualisasi Hasil Prediksi")
                    status_counts = df_predict['STATUS KEBUTUHAN'].value_counts().reset_index()
                    fig = px.bar(status_counts, x='STATUS KEBUTUHAN', y='count', color='STATUS KEBUTUHAN',
                                 color_discrete_map={'Restok': "#AF0000", 'Tidak Restok': "#53055A"}, text='count',
                                 labels={'count': 'Jumlah Item', 'STATUS KEBUTUHAN': 'Status Kebutuhan'})
                    fig.update_layout(height=500, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("---")
                    st.subheader("Unduh Laporan Hasil Prediksi")
                    col1, col2, col3 = st.columns(3)
                    col1.download_button("Unduh .CSV", df_predict.to_csv(index=False).encode('utf-8'), "hasil_prediksi.csv", "text/csv", use_container_width=True, type="secondary")
                    col2.download_button("Unduh .XLSX", to_excel(df_predict), "hasil_prediksi.xlsx", use_container_width=True, type="secondary")
                    col3.download_button("Unduh .PDF", to_pdf(df_predict, "Laporan Prediksi Kebutuhan Stok Obat"), "hasil_prediksi.pdf", use_container_width=True, type="secondary")
                    st.markdown('</div>', unsafe_allow_html=True)

                with st.container(border=False):
                     st.markdown('<div class="card">', unsafe_allow_html=True)
                     st.subheader("Detail Data Hasil Prediksi")
                     tab1, tab2, tab3 = st.tabs(["Semua Hasil", "Item Perlu Restok", "Item Stok Cukup"])
                     with tab1: st.dataframe(df_predict, use_container_width=True)
                     with tab2: st.dataframe(df_predict[df_predict['STATUS KEBUTUHAN'] == 'Restok'], use_container_width=True)
                     with tab3: st.dataframe(df_predict[df_predict['STATUS KEBUTUHAN'] == 'Tidak Restok'], use_container_width=True)
                     
                     with st.expander("Lihat Visualisasi Pohon Keputusan (Decision Tree)"):
                        fig, ax = plt.subplots(figsize=(25, 15))
                        plot_tree(model, feature_names=model_features, class_names=le.classes_, filled=True, rounded=True, fontsize=10)
                        st.pyplot(fig)
                     st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
            st.warning("Pastikan kolom yang Anda petakan berisi data numerik yang valid.")


# --- TAB 2: EVALUASI MODEL ---
with tab_evaluasi:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Evaluasi Kinerja Model")
    st.info("Unggah file (.csv/.xlsx) yang berisi data historis termasuk **label asli** ('Restok'/'Tidak Restok') untuk mengukur kinerja model.")
    
    eval_file = st.file_uploader("Unggah file evaluasi", type=['csv', 'xlsx'], key="eval_uploader", label_visibility="collapsed")

    if eval_file and model:
        try:
            df_eval_input = pd.read_csv(eval_file) if eval_file.name.endswith('.csv') else pd.read_excel(eval_file)
            st.success(f"File '{eval_file.name}' berhasil diunggah. Terdapat {len(df_eval_input)} baris.")

            st.markdown("##### Pemetaan Kolom untuk Evaluasi")
            col1, col2, col3 = st.columns(3)
            available_cols_eval = ['-'] + df_eval_input.columns.tolist()
            eval_col_stok_awal = col1.selectbox("Kolom Stok Awal:", available_cols_eval, key="eval_stok")
            eval_col_terjual = col2.selectbox("Kolom Terjual:", available_cols_eval, key="eval_terjual")
            eval_col_target = col3.selectbox("Kolom Label Asli (Target):", available_cols_eval, key="eval_target")

            if eval_col_stok_awal != '-' and eval_col_terjual != '-' and eval_col_target != '-':
                if st.button("Lakukan Evaluasi Model", key="evaluate_model_btn", use_container_width=True):
                    df_eval = df_eval_input.copy()
                    df_eval['STOK AWAL'] = pd.to_numeric(df_eval[eval_col_stok_awal], errors='coerce')
                    df_eval['TERJUAL'] = pd.to_numeric(df_eval[eval_col_terjual], errors='coerce')
                    
                    if df_eval['STOK AWAL'].isnull().any() or df_eval['TERJUAL'].isnull().any():
                        st.error("Data fitur ('Stok Awal' atau 'Terjual') mengandung nilai non-numerik.")
                    else:
                        df_eval['SISA STOK'] = df_eval['STOK AWAL'] - df_eval['TERJUAL']
                        X_eval = df_eval[model_features]
                        y_true_text = df_eval[eval_col_target]

                        if not all(item in le.classes_ for item in y_true_text.unique()):
                            st.error(f"Kolom target berisi nilai yang tidak dikenal. Harap gunakan hanya {le.classes_.tolist()}.")
                        else:
                            y_pred_text = le.inverse_transform(model.predict(X_eval))
                            st.subheader("Hasil Metrik Evaluasi", anchor=False)

                            # === BAGIAN YANG DIPERBARUI ===
                            # Hitung metrik
                            accuracy = accuracy_score(y_true_text, y_pred_text)
                            precision = precision_score(y_true_text, y_pred_text, average='weighted', zero_division=0)
                            recall = recall_score(y_true_text, y_pred_text, average='weighted', zero_division=0)

                            # Buat DataFrame untuk plot
                            metrics_data = {
                                'Metrik': ['Akurasi', 'Presisi', 'Recall'],
                                'Nilai': [accuracy, precision, recall]
                            }
                            df_metrics = pd.DataFrame(metrics_data)

                            # Buat diagram batang
                            fig_metrics = px.bar(
                                df_metrics,
                                x='Metrik',
                                y='Nilai',
                                title="Visualisasi Metrik Evaluasi",
                                text=[f'{val:.2%}' for val in df_metrics['Nilai']], # Format teks sebagai persen
                                color='Metrik',
                                color_discrete_map={
                                    'Akurasi': "#000DC0", 'Presisi': "#FFA600", 'Recall': "#FF0037"}
                            )
                            fig_metrics.update_layout(
                                yaxis_title='Skor',
                                xaxis_title=None,
                                showlegend=False,
                                yaxis=dict(range=[0, 1.05], tickformat=".0%") # Atur sumbu Y ke format persen
                            )
                            st.plotly_chart(fig_metrics, use_container_width=True)
                            
        except Exception as e:
            st.error(f"Gagal memproses file evaluasi. Error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)