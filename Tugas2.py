# ==========================================================
# KASUS A: Diagnosis Bias-Variance dengan Decision Tree
# depth = 1, 4, None (tak terbatas)
# ==========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score, precision_score

# ----------------------------------------------------------
# 1. BUAT DATASET
# ----------------------------------------------------------
rng = np.random.default_rng(42)
n = 1500

df = pd.DataFrame({
    "ipk": np.clip(rng.normal(3.0, 0.5, n), 1.0, 4.0),
    "ips_terakhir": np.clip(rng.normal(3.0, 0.6, n), 0.5, 4.0),
    "sks_lulus": rng.integers(40, 90, n),
    "jumlah_mk_mengulang": rng.poisson(1.0, n),
    "jumlah_nilai_d_e": rng.poisson(1.5, n),
    "kehadiran_persen": np.clip(rng.normal(80, 15, n), 20, 100),
    "login_lms_per_minggu": rng.poisson(8, n),
    "tugas_terlambat_persen": np.clip(rng.normal(20, 15, n), 0, 100),
    "penerima_beasiswa": rng.integers(0, 2, n),
    "bekerja_paruh_waktu": rng.integers(0, 2, n),
})

# Hubungan non-linear (interaksi) + noise agar depth=1 tidak cukup
# dan depth=None mudah menghafal noise
skor = (
    -1.6 * (df["ipk"] - 3.0)
    - 0.8 * (df["ips_terakhir"] - 3.0)
    + 0.35 * df["jumlah_mk_mengulang"]
    + 0.25 * df["jumlah_nilai_d_e"]
    - 0.03 * (df["kehadiran_persen"] - 80)
    + 0.02 * (df["tugas_terlambat_persen"] - 20)
    + 0.9 * ((df["ipk"] < 2.7) & (df["kehadiran_persen"] < 70))   # interaksi
    + 0.4 * df["bekerja_paruh_waktu"]
    - 0.3 * df["penerima_beasiswa"]
    - 1.2
    + rng.normal(0, 1.0, n)                                        # noise
)
prob = 1 / (1 + np.exp(-skor))
df["tidak_lulus_tepat_waktu"] = (rng.random(n) < prob).astype(int)

print("Distribusi target:")
print(df["tidak_lulus_tepat_waktu"].value_counts(normalize=True).round(3), "\n")

# ----------------------------------------------------------
# 2. SPLIT + STRATIFY
# ----------------------------------------------------------
X = df.drop(columns="tidak_lulus_tepat_waktu")
y = df["tidak_lulus_tepat_waktu"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Proporsi kelas 1 -> train:", round(y_train.mean(), 3),
      "| test:", round(y_test.mean(), 3), "\n")

# ----------------------------------------------------------
# 3. LATIH DEPTH 1, 4, TAK TERBATAS
# ----------------------------------------------------------
depths = {"depth=1": 1, "depth=4": 4, "depth=∞ (None)": None}
hasil = {}

for nama, d in depths.items():
    model = DecisionTreeClassifier(max_depth=d, class_weight="balanced",
                                   random_state=42)
    model.fit(X_train, y_train)
    pred_tr = model.predict(X_train)
    pred_te = model.predict(X_test)
    hasil[nama] = {
        "model": model,
        "f1_train": f1_score(y_train, pred_tr),
        "f1_test": f1_score(y_test, pred_te),
        "recall_test": recall_score(y_test, pred_te),
        "precision_test": precision_score(y_test, pred_te),
        "depth_aktual": model.get_depth(),
        "n_daun": model.get_n_leaves(),
    }

tabel = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "model"}
                      for k, v in hasil.items()}).T
tabel["gap"] = tabel["f1_train"] - tabel["f1_test"]
print("RINGKASAN HASIL")
print(tabel.round(3), "\n")

# ----------------------------------------------------------
# 4. PLOT LEARNING CURVE
# ----------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ukuran = np.linspace(0.1, 1.0, 8)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
lc_ringkas = {}

for ax, (nama, d) in zip(axes, depths.items()):
    model = DecisionTreeClassifier(max_depth=d, class_weight="balanced",
                                   random_state=42)
    sizes, tr_sc, va_sc = learning_curve(
        model, X_train, y_train, cv=cv, train_sizes=ukuran,
        scoring="f1", n_jobs=-1, random_state=42
    )
    tr_m, tr_s = tr_sc.mean(axis=1), tr_sc.std(axis=1)
    va_m, va_s = va_sc.mean(axis=1), va_sc.std(axis=1)
    lc_ringkas[nama] = (tr_m[-1], va_m[-1])

    ax.plot(sizes, tr_m, "o-", label="Train")
    ax.plot(sizes, va_m, "o-", label="Validasi (CV)")
    ax.fill_between(sizes, tr_m - tr_s, tr_m + tr_s, alpha=0.15)
    ax.fill_between(sizes, va_m - va_s, va_m + va_s, alpha=0.15)
    ax.set_title(nama)
    ax.set_xlabel("Jumlah data latih")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

axes[0].set_ylabel("F1 (kelas berisiko)")
plt.suptitle("Learning Curve Decision Tree - Prediksi Mahasiswa Berisiko", y=1.02)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 5. DIAGNOSIS (otomatis dari angka)
# ----------------------------------------------------------
def diagnosis(nama, f1_tr, f1_va, ambang_rendah=0.65, ambang_gap=0.10):
    gap = f1_tr - f1_va
    if f1_tr < ambang_rendah and f1_va < ambang_rendah and gap < ambang_gap:
        status = "UNDERFITTING (bias tinggi)"
        alasan = ("Skor train dan validasi sama-sama rendah dan berdekatan. "
                  "Model terlalu sederhana untuk menangkap pola data.")
        saran = ("Naikkan kompleksitas (max_depth lebih besar), tambah fitur "
                 "interaksi, atau pakai model yang lebih ekspresif.")
    elif gap >= ambang_gap:
        status = "OVERFITTING (varians tinggi)"
        alasan = ("Skor train jauh di atas validasi. Model menghafal data latih "
                  "termasuk noise, sehingga gagal menggeneralisasi.")
        saran = ("Batasi kedalaman, atur min_samples_leaf / ccp_alpha, "
                 "tambah data, atau pakai ensemble (Random Forest).")
    else:
        status = "FIT CUKUP BAIK (seimbang)"
        alasan = ("Skor train dan validasi tinggi dan gap kecil. "
                  "Bias dan varians relatif seimbang.")
        saran = ("Pertahankan konfigurasi ini; lanjut tuning threshold "
                 "untuk mengejar recall.")
    return (f"\n[{nama}]\n"
            f"  F1 train = {f1_tr:.3f} | F1 validasi = {f1_va:.3f} | gap = {gap:.3f}\n"
            f"  Status : {status}\n"
            f"  Alasan : {alasan}\n"
            f"  Saran  : {saran}")

print("=" * 60)
print("DIAGNOSIS")
print("=" * 60)
for nama, (tr, va) in lc_ringkas.items():
    print(diagnosis(nama, tr, va))

# Model terbaik = F1 validasi tertinggi
terbaik = max(lc_ringkas, key=lambda k: lc_ringkas[k][1])
print("\n" + "=" * 60)
print(f"KESIMPULAN: model terbaik berdasarkan F1 validasi -> {terbaik}")
print("Konfirmasi di test set:",
      round(hasil[terbaik]["f1_test"], 3), "(F1) |",
      round(hasil[terbaik]["recall_test"], 3), "(recall)")
print("=" * 60)