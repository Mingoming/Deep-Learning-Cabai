# Klasifikasi Daun Cabai

Implementasi eksperimen ada di **Klasifikasi_Daun_Cabai.ipynb**. Notebook memakai satu pipeline untuk MobileNetV3-Small, ResNet-50, dan EfficientNet-B0.

## Menjalankan notebook di VS Code

1. Buka notebook dan pilih kernel **Python 3.10 (Deep-Learning-Cabai .venv)**.
2. Gunakan **Run All** dengan `JALANKAN_BASELINE = False` dan `JALANKAN_KOMPARASI = False`. Notebook membaca artefak tersimpan tanpa training ulang, lalu mengevaluasi tiga checkpoint terbaik pada validation. Hasilnya mencakup tabel riwayat dan metrik, laporan per kelas, confusion matrix, contoh salah klasifikasi, jumlah parameter, ukuran checkpoint, waktu inferensi, dan grafik.
3. Untuk menjalankan komparasi baru, ubah `JALANKAN_KOMPARASI = True` lalu jalankan cell training komparasi saja. ResNet-50 dan EfficientNet-B0 dilatih berurutan dan hasil disimpan pada direktori bertimestamp baru. Kembalikan nilainya ke `False`, lalu jalankan notebook untuk menampilkan artefak tersimpan.
4. `JALANKAN_BASELINE` hanya diperlukan untuk mengulang MobileNetV3-Small; biarkan `False` untuk komparasi model yang sudah tersimpan.

Bobot ImageNet TorchVision disimpan di `data/processed/torch_hub/` agar dapat digunakan kembali.

## Data dan metodologi

- Persiapan dataset satu kali berada di `scripts/prepare_data.py`. Script menolak menimpa artefak yang sudah tersedia; notebook membaca `data/extracted/` dan `data/processed/split_manifest.csv` tanpa mengekstrak atau mengaudit ulang ZIP.
- Arsip asli di `data/` dan sumber Progress I di `docs/` adalah read-only.
- Keputusan pemilik proyek dicatat di `docs/DL-01-DL-02-decisions.md`; PPT Progress I menjadi acuan jika bertentangan dengan laporan lama.
- Eksperimen memakai input RGB 224 x 224, normalisasi ImageNet, split group-aware 70:15:15 dengan manifest seed 42/43/44, augmentasi dan class weight train-only, serta validation Macro-F1 untuk pemilihan checkpoint.
- Konfigurasi pilot: classifier head 3 epoch dengan LR 1e-3, lalu fine-tuning seluruh model 8 epoch dengan LR 1e-4; batch 32, AdamW, weight decay 1e-4, label smoothing 0,1, dan cosine annealing. Detail setiap run tersimpan di `experiment_config.json`.
- Test set disimpan untuk evaluasi akhir dan tidak dipakai memilih checkpoint pada training ini.

## Pemeriksaan

Di VS Code, pilih kernel **Python 3.10 (Deep-Learning-Cabai .venv)** lalu jalankan **Run All**. Dengan kedua flag training `False`, notebook menampilkan hasil tersimpan dan menjalankan evaluasi validation tanpa training ulang. Test set tidak dievaluasi.

Untuk pemeriksaan kode pipeline:

    .\.venv\Scripts\python.exe -m unittest discover -s tests -v

Jangan tambahkan dataset ke Git.
