
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px # <--- PENTING: Untuk membuat grafik horizontal yang tidak miring

# Menggunakan cache untuk model agar hanya di-load sekali
@st.cache_resource
def load_model():
    """
    Memuat pipeline model dan preprocessor yang sudah disimpan.
    """
    try:
        # Ganti dengan nama file model Anda yang benar jika berbeda
        model_data = joblib.load('svm_model.joblib')
        return model_data
    except FileNotFoundError:
        return None

def main():
    st.set_page_config(page_title="Prediksi ASD pada Anak", layout="wide")
    st.title("👶 Prediksi Spektrum Autisme (ASD) pada Anak")

    # Load model
    model_data = load_model()

    if model_data is None:
        st.error("File model 'svm_model.joblib' tidak ditemukan. Pastikan file tersebut ada di repositori GitHub Anda bersama 'app.py'.")
        return

    # Ekstrak objek yang disimpan
    model = model_data['model']
    le_y = model_data['le_y']
    le_sex = model_data['le_sex']
    le_jaundice = model_data['le_jaundice']
    le_family_asd = le_family_asd = model_data['le_family_asd']
    training_columns = model_data['training_columns']

    st.write("Aplikasi ini memprediksi kemungkinan ASD pada balita menggunakan model SVM Linear yang sudah dilatih.")

    # --- Inisialisasi Alur Halaman (State Management) ---
    if "page" not in st.session_state:
        st.session_state.page = 1
        st.session_state.demographics = {}
        st.session_state.results = {}

    # --- Definisi Opsi (ditaruh di luar agar rapi) ---
    questions = {
        "A1": "Apakah anak Anda melihat Anda ketika Anda memanggil namanya?",
        "A2": "Seberapa mudah bagi Anda untuk mendapatkan kontak mata dengan anak Anda?",
        "A3": "Apakah anak Anda menunjuk untuk menunjukkan bahwa ia menginginkan sesuatu?",
        "A4": "Apakah anak Anda menunjuk untuk berbagi minat dengan Anda?",
        "A5": "Apakah anak Anda berpura-pura (misalnya, merawat boneka)?",
        "A6": "Apakah anak Anda mengikuti ke mana Anda melihat?",
        "A7": "Jika Anda sedih, apakah anak Anda menunjukkan tanda-tanda ingin menghibur?",
        "A8": "Apakah Anda akan mendeskripsikan kata-kata pertama anak Anda sebagai 'biasa'?",
        "A9": "Apakah anak Anda menggunakan gerakan sederhana (misalnya, melambaikan tangan)?",
        "A10": "Apakah anak Anda menatap ke hal yang tidak ada tanpa tujuan yang jelas?"
    }

    # Opsi jawaban baru
    options_list = ["Selalu", "Biasanya", "Kadang-kadang", "Jarang", "Tidak pernah"]

    # Logika skoring A1-A9 (dibalik): 1 poin jika "Kadang-kadang", "Jarang", "Tidak pernah"
    mapping_A1_A9 = {
        "Selalu": 0, "Biasanya": 0, "Kadang-kadang": 1, "Jarang": 1, "Tidak pernah": 1
    }

    # Logika skoring A10: 1 poin jika "Selalu", "Biasanya", "Kadang-kadang"
    mapping_A10 = {
        "Selalu": 1, "Biasanya": 1, "Kadang-kadang": 1, "Jarang": 0, "Tidak pernah": 0
    }

    # Buat pemetaan untuk tampilan "Iya" / "Tidak"
    display_map_yes_no = {"Yes": "Iya", "No": "Tidak"}

    # Ambil kelas asli dari encoder yang disimpan
    sex_options = le_sex.classes_
    jaundice_options = le_jaundice.classes_
    family_options = le_family_asd.classes_

    # Opsi Etnis dan Pengisi Tes (hardcode dari data Anda)
    ethnicity_options = [
        'Timur Tengah', 'Eropa Kulit Putih', 'Hispanik', 'Kulit Hitam', 'Asia',
        'Asian Selatan', 'Indian Asli', 'Yang lain', 'Latin', 'Campuran', 'Pasifik'
    ]
    tester_options = [
        'Anggota Keluarga', 'Tenaga kesehatan', 'Diri sendiri', 'Yang lain'
    ]

    # --- Halaman 1: Latar Belakang Anak ---
    if st.session_state.page == 1:
        st.header("Langkah 1: Latar Belakang Kondisi Anak")
        st.write("Silakan isi data demografis anak terlebih dahulu.")

        with st.form("form_langkah_1"):
            # Gunakan st.columns untuk layout yang lebih rapi
            col1, col2 = st.columns(2)
            with col1:
                age_mons = st.number_input("Usia (dalam bulan)", min_value=12, max_value=36, value=24)
                sex = st.selectbox("Jenis Kelamin", options=sex_options, format_func=lambda x: "Perempuan" if x == 'F' else "Laki-laki")
                jaundice = st.selectbox(
                    "Riwayat Penyakit Kuning (Jaundice)",
                    options=jaundice_options,
                    format_func=lambda x: display_map_yes_no.get(x, x)
                )
            with col2:
                family_asd = st.selectbox(
                    "Riwayat ASD dalam Keluarga",
                    options=family_options,
                    format_func=lambda x: display_map_yes_no.get(x, x)
                )
                ethnicity = st.selectbox("Etnis", options=ethnicity_options)
                who_completed = st.selectbox("Siapa yang Mengisi Tes?", options=tester_options)

            # Tombol Lanjut
            submitted1 = st.form_submit_button("Lanjut ke Kuesioner")

            if submitted1:
                # Simpan data dari form ini ke session state
                st.session_state.demographics = {
                    "Age_Mons": age_mons,
                    "Sex": sex,
                    "Jaundice": jaundice,
                    "Family_mem_with_ASD": family_asd,
                    "Ethnicity": ethnicity,
                    "Who completed the test": who_completed
                }
                # Pindah ke halaman 2
                st.session_state.page = 2
                st.rerun() # Memaksa Streamlit untuk menggambar ulang halaman baru

    # --- Halaman 2: Kuesioner ---
    elif st.session_state.page == 2:
        st.header("Langkah 2: Kuesioner Q-Chat-10")
        st.write("Jawablah pertanyaan berikut berdasarkan perilaku anak Anda.")

        with st.form("form_langkah_2"):
            question_inputs = {}
            for key, value in questions.items():
                # --- Menggunakan st.radio (horizontal) ---
                selected_option = st.radio(
                    f"**{key}:** {value}",
                    options=options_list,
                    key=key,
                    horizontal=True
                )

                # Terapkan logika skoring yang sesuai
                if key == "A10":
                    question_inputs[key] = mapping_A10[selected_option]
                else:
                    question_inputs[key] = mapping_A1_A9[selected_option]

            # Tombol Prediksi
            submitted2 = st.form_submit_button("Prediksi Kemungkinan ASD")

            if submitted2:
                # Gabungkan data dari langkah 1 (disimpan di state) dan langkah 2 (dari form ini)
                input_data = st.session_state.demographics.copy()
                input_data.update(question_inputs)

                # --- Logika Prediksi ---
                try:
                    input_df = pd.DataFrame([input_data])
                    input_df['Sex'] = le_sex.transform(input_df['Sex'])
                    input_df['Jaundice'] = le_jaundice.transform(input_df['Jaundice'])
                    input_df['Family_mem_with_ASD'] = le_family_asd.transform(input_df['Family_mem_with_ASD'])

                    input_df_encoded = pd.get_dummies(input_df, columns=["Ethnicity", "Who completed the test"])
                    input_df_final = input_df_encoded.reindex(columns=training_columns, fill_value=0)

                    prediction = model.predict(input_df_final)[0]
                    prediction_proba = model.predict_proba(input_df_final)[0]
                    prediction_label = le_y.inverse_transform([prediction])[0]

                    # Simpan hasil ke session state
                    st.session_state.results = {
                        "label": prediction_label,
                        "proba": prediction_proba,
                        "classes": le_y.classes_
                    }

                    # Pindah ke halaman 3 (Hasil)
                    st.session_state.page = 3
                    st.rerun()

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

        st.divider()
        if st.button("Kembali ke Langkah 1"):
            st.session_state.page = 1
            st.rerun()

    # --- Halaman 3: Hasil Prediksi ---
    elif st.session_state.page == 3:
        st.header("Hasil Prediksi Model (SVM Linear)")

        results = st.session_state.results
        prediction_label = results["label"]
        prediction_proba = results["proba"]
        classes = results["classes"]

        # Dapatkan indeks untuk kelas 'Yes' dan 'No'
        try:
            index_yes = list(classes).index('Yes')
            index_no = list(classes).index('No')
        except ValueError:
            st.error("Kelas prediksi tidak valid. Harap periksa model Anda.")
            return

        # Ambil probabilitas berdasarkan indeks
        proba_yes = prediction_proba[index_yes]
        proba_no = prediction_proba[index_no]

        # Tampilkan hasil utama
        if prediction_label == "Yes":
            st.error(f"**Prediksi: Terdeteksi Tanda-Tanda ASD (Iya)**", icon="⚠️")
            st.write(f"Model memiliki keyakinan **{proba_yes:.2%}** bahwa terdapat tanda-tanda ASD.")
            st.info("Harap dicatat bahwa ini **bukan diagnosis medis**. Silakan berkonsultasi dengan profesional kesehatan anak atau psikolog untuk evaluasi lebih lanjut.")
        else:
            st.success(f"**Prediksi: Tidak Terdeteksi Tanda-Tanda ASD (Tidak)**", icon="✅")
            st.write(f"Model memiliki keyakinan **{proba_no:.2%}** bahwa tidak terdapat tanda-tanda ASD.")

        # --- SOLUSI GRAFIK DENGAN PLOTLY (Vertikal, label tegak) ---
        st.subheader("Probabilitas Hasil Prediksi")

        # 1. Siapkan DataFrame untuk Plotly (Vertikal)
        chart_data_plotly = pd.DataFrame({
            # Kategori pada sumbu X, nilai probabilitas pada sumbu Y
            'Kategori': ['Tidak (Non-ASD)', 'Iya (Tanda ASD)'],
            'Probabilitas': [proba_no, proba_yes]
        })

        # 2. Buat Grafik Batang Vertikal menggunakan Plotly Express
        fig = px.bar(
            chart_data_plotly,
            x='Kategori',       # Label yang ditampilkan (sekarang di sumbu X)
            y='Probabilitas',   # Nilai yang divisualisasikan
            orientation='v',    # Orientasi vertikal
            height=400,
            color='Kategori',
            color_discrete_map={
                'Tidak (Non-ASD)': '#4CAF50', # Hijau
                'Iya (Tanda ASD)': '#F44336'   # Merah
            },
            title=''
        )

        # 3. Atur layout untuk memaksa label tetap horizontal (tickangle=0)
        fig.update_xaxes(
            tickangle=0,            # MEMAKSA LABEL X-AXIS TETAP HORIZONTAL (Tegak)
            title_text="",          # Hilangkan judul sumbu X
        )

        fig.update_layout(
            yaxis_title="Probabilitas (%)",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(b=100) # Menambah margin bawah untuk memberi ruang label yang panjang
        )

        # 4. Tampilkan grafik di Streamlit
        st.plotly_chart(fig, use_container_width=True)
        # --- END SOLUSI GRAFIK DENGAN PLOTLY ---

        st.divider()
        if st.button("Isi Ulang Kuesioner (Mulai Lagi)"):
            # Reset state untuk kembali ke halaman 1
            st.session_state.page = 1
            st.session_state.demographics = {}
            st.session_state.results = {}
            st.rerun()

if __name__ == "__main__":
    main()
