# MKYZ Deployment & Rebuild Guide

## 1. PyPI'a Yeni Versiyonu Yükle

Aşağıdaki komutları PowerShell üzerinde sırasıyla çalıştırın:

```powershell
# Eski build dosyalarını temizle
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue

# Gerekli araçları kur (eğer yoksa)
pip install build twine

# Paketi oluştur (sdist ve wheel)
python -m build

# PyPI'a yükle (PyPI kullanıcı adı ve şifrenizi veya API Token'ınızı soracaktır)
twine upload dist/*
```

## 2. ReadTheDocs Güncellemesi

Değişiklikleri GitHub'a yolladığınızda ReadTheDocs otomatik olarak güncellenecektir:

```bash
# Değişiklikleri hazırla
git add .

# Versiyon güncelleme mesajıyla commit et
git commit -m "Bump version to 0.2.0 and switch docs to MkDocs"

# GitHub'a pushla
git push
```

> **Not:** GitHub'a pushladığınızda ReadTheDocs otomatik olarak yeni MkDocs yapısını algılayıp siteyi güncelleyecektir.
