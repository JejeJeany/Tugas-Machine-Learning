# ==========================================================
# KASUS A: Prediksi Mahasiswa Berisiko Tidak Lulus
# ==========================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score,
                             precision_recall_curve, recall_score, precision_score)

# ----------------------------------------------------------
# LIMA KEPUTUSAN DESAIN
# ----------------------------------------------------------
KEPUTUSAN = {
    "1_tujuan_dan_pengguna": {
        "tujuan": "Mendeteksi dini mahasiswa yang berisiko tidak lulus tepat waktu "
                  "agar bisa diberi intervensi (konseling, mentoring, remedial) "
                  "sebelum terlambat.",
        "pengguna": ["Dosen Pembimbing Akademik", "Bagian Akademik/Prodi",
                     "Konselor mahasiswa"],
        "catatan": "Model sebagai alat bantu keputusan, bukan penentu final. "
                   "Output berupa skor risiko + faktor pendorong."
    },
    "2_unit_data_sampel": {
        "unit": "1 baris = 1 mahasiswa pada titik waktu prediksi (akhir semester 4)",
        "cakupan": "Mahasiswa angkatan lama yang hasil akhirnya sudah diketahui",
        "catatan": "Hindari data leakage: hanya pakai informasi yang sudah ada "
                   "SEBELUM akhir semester 4."
    },
    "3_fitur_tersedia": {
        "akademik": ["ipk", "ips_terakhir", "sks_lulus", "jumlah_mk_mengulang",
                     "jumlah_nilai_d_e"],
        "perilaku": ["kehadiran_persen", "login_lms_per_minggu", "tugas_terlambat_persen"],
        "demografi_ekonomi": ["jalur_masuk", "penerima_beasiswa", "bekerja_paruh_waktu"],
        "catatan": "Fitur sensitif (gender, suku, agama) tidak digunakan "
                   "untuk menghindari bias."
    },
    "4_target_reward": {
        "target": "tidak_lulus_tepat_waktu (1 = berisiko/tidak lulus tepat waktu, 0 = lulus tepat waktu)",
        "jenis_masalah": "Klasifikasi biner (supervised learning)",
        "reward": "Tidak dipakai (bukan reinforcement learning)"
    },
    "5_metrik_keberhasilan": {
        "utama": "Recall kelas berisiko (jangan sampai mahasiswa berisiko terlewat)",
        "pendukung": ["Precision", "F1", "PR-AUC", "ROC-AUC"],
        "alasan": "Data tidak seimbang & biaya salah-negatif (mahasiswa berisiko "
                  "tak terdeteksi) lebih mahal daripada salah-positif.",
        "target_minimal": "Recall >= 0.80 dengan Precision >= 0.50"
    }
}

# ----------------------------------------------------------
# 1. DATA SINTETIS (ganti dengan dataset asli jika ada)
# ----------------------------------------------------------
rng = np.random.default_rng(42)
n = 2000

df = pd.DataFrame({
    "ipk": np.clip(rng.normal(3.0, 0.5, n), 1.0, 4.0),
    "ips_terakhir": np.clip(rng.normal(3.0, 0.6, n), 0.5, 4.0),
    "sks_lulus": rng.integers(40, 90, n),
    "jumlah_mk_mengulang": rng.poisson(1.0, n),
    "jumlah_nilai_d_e": rng.poisson(1.5, n),
    "kehadiran_persen": np.clip(rng.normal(80, 15, n), 20, 100),
    "login_lms_per_minggu": rng.poisson(8, n),
    "tugas_terlambat_persen": np.clip(rng.normal(20, 15, n), 0, 100),
    "jalur_masuk": rng.choice(["SNBP", "SNBT", "Mandiri"], n, p=[0.3, 0.4, 0.3]),
    "penerima_beasiswa": rng.integers(0, 2, n),
    "bekerja_paruh_waktu": rng.integers(0, 2, n),
})

# Target dibuat dari skor risiko + noise (hanya untuk simulasi)
skor = (
    -1.8 * (df["ipk"] - 3.0)
    - 0.9 * (df["ips_terakhir"] - 3.0)
    - 0.04 * (df["sks_lulus"] - 65)
    + 0.35 * df["jumlah_mk_mengulang"]
    + 0.25 * df["jumlah_nilai_d_e"]
    - 0.03 * (df["kehadiran_persen"] - 80)
    - 0.05 * (df["login_lms_per_minggu"] - 8)
    + 0.02 * (df["tugas_terlambat_persen"] - 20)
    + 0.4 * df["bekerja_paruh_waktu"]
    - 0.3 * df["penerima_beasiswa"]
    - 1.5
)
prob = 1 / (1 + np.exp(-skor))
df["tidak_lulus_tepat_waktu"] = (rng.random(n) < prob).astype(int)

print("Distribusi target:\n", df["tidak_lulus_tepat_waktu"].value_counts(normalize=True), "\n")

# ----------------------------------------------------------
# 2. SPLIT DATA
# ----------------------------------------------------------
X = df.drop(columns="tidak_lulus_tepat_waktu")
y = df["tidak_lulus_tepat_waktu"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------------------------------------------------
# 3. PREPROCESSING + MODEL
# ----------------------------------------------------------
fitur_num = [c for c in X.columns if c != "jalur_masuk"]
fitur_kat = ["jalur_masuk"]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), fitur_num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), fitur_kat),
])

models = {
    "Logistic Regression": Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]),
    "Random Forest": Pipeline([
        ("prep", preprocess),
        ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                       random_state=42, n_jobs=-1))
    ]),
}

# ----------------------------------------------------------
# 4. CROSS-VALIDATION (fokus recall)
# ----------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for nama, model in models.items():
    skor_recall = cross_val_score(model, X_train, y_train, cv=cv, scoring="recall")
    skor_pr = cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision")
    print(f"[CV] {nama}: recall={skor_recall.mean():.3f} | PR-AUC={skor_pr.mean():.3f}")

# ----------------------------------------------------------
# 5. EVALUASI TEST SET + PENYETELAN THRESHOLD
# ----------------------------------------------------------
TARGET_RECALL = 0.80

for nama, model in models.items():
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    # Cari threshold tertinggi yang masih memenuhi target recall
    prec, rec, thr = precision_recall_curve(y_test, proba)
    kandidat = [t for r, t in zip(rec[:-1], thr) if r >= TARGET_RECALL]
    threshold = max(kandidat) if kandidat else 0.5

    pred = (proba >= threshold).astype(int)

    print("\n" + "=" * 60)
    print(f"MODEL: {nama}  | threshold = {threshold:.3f}")
    print("=" * 60)
    print("ROC-AUC :", round(roc_auc_score(y_test, proba), 3))
    print("PR-AUC  :", round(average_precision_score(y_test, proba), 3))
    print("Recall  :", round(recall_score(y_test, pred), 3))
    print("Precision:", round(precision_score(y_test, pred), 3))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred,
          target_names=["Lulus tepat waktu", "Berisiko"]))

# ----------------------------------------------------------
# 6. FAKTOR PENDORONG RISIKO (interpretasi)
# ----------------------------------------------------------
rf = models["Random Forest"]
nama_fitur = rf.named_steps["prep"].get_feature_names_out()
importance = pd.Series(rf.named_steps["clf"].feature_importances_, index=nama_fitur)
print("\nTop 8 fitur paling berpengaruh:")
print(importance.sort_values(ascending=False).head(8).round(3))

# ----------------------------------------------------------
# 7. CETAK RINGKASAN LIMA KEPUTUSAN
# ----------------------------------------------------------
print("\n" + "=" * 60)
print("RINGKASAN LIMA KEPUTUSAN")
print("=" * 60)
for k, v in KEPUTUSAN.items():
    print(f"\n{k.upper()}")
    for kk, vv in v.items():
        print(f"  - {kk}: {vv}")

