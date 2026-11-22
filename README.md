# Türkçe SMS Spam Tespiti

Doğal Dil İşleme dersi proje çalışması. Türkçe SMS mesajlarını **spam** ve **normal** olarak sınıflandıran, derste görülen yöntemlerle (sayma vektörleri, TF-IDF, Markov dil modeli, Naive Bayes, dengesiz veri metrikleri, kosinüs benzerliği) geliştirilmiş bir proje.

Tüm kod ve çıktılar [`sms_spam_tespiti.ipynb`](sms_spam_tespiti.ipynb) notebook'unda.

## Veri seti

**Turkish SMS Collection** (Kaggle): https://www.kaggle.com/datasets/onurkarasoy/turkish-sms-collection

* Türkiye'nin farklı bölgelerinde yaşayan, farklı yaş gruplarından insanlardan toplanmış, **spam / normal** etiketli Türkçe SMS'ler
* 4.751 mesaj (2.536 spam, 2.215 normal). Tekrarlanan 11 mesaj atıldıktan sonra **4.740 mesaj** (2.526 spam, 2.214 normal)
* Sütunlar: `Message` (mesaj), `Group` (1 = spam, 2 = normal), `GroupText` (Spam / Normal)
* Kaynak çalışma: Karasoy, O., Ballı, S. *Spam SMS Detection for Turkish Language with Deep Text Analysis and Deep Learning Methods.* Arab J Sci Eng (2021).

## Kullanılan yöntemler

| Bölüm | Yöntem | Dersteki karşılığı |
|---|---|---|
| 1 | Keşifsel veri analizi (sınıf dağılımı, mesaj uzunlukları) | LAB0 |
| 2 | Türkçe metin temizleme, tokenization, özel token'lar (`<url>`, `<tel>`, `<para>`, `<sayi>`) | Ders 1, LAB1 |
| 4 | **Sıfırdan yazılmış** Markov dil modeli tabanlı Bayes sınıflandırıcı (log olasılık, add-one yumuşatma, `<unk>`) | Ders 4–5, LAB3–4, toy_bayes |
| 5 | Sayma vektörleri ve TF-IDF + Multinomial Naive Bayes, unigram/bigram karşılaştırması | Ders 2–3, LAB1–2, LAB5 |
| 6 | Dengesiz veri senaryosu: accuracy, precision, recall, F1, confusion matrix | Ders 6, LAB5 |
| 7 | Kosinüs benzerliği ile benzer mesaj ve spam kampanyası bulma | Docsim |
| 8 | Markov modeliyle yapay spam SMS üretme | Ders 4, LAB3 |

## Sonuçlar

### Dengeli veri seti (test kümesi: 948 mesaj, %80 / %20 stratified ayrım)

| Model | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) |
|---|---|---|---|---|
| Markov-Bayes (sıfırdan) | 0.9673 | 0.9422 | 1.0000 | 0.9702 |
| **MNB + Count (unigram)** | **0.9810** | **0.9656** | **1.0000** | **0.9825** |
| MNB + Count (uni+bigram) | 0.9705 | 0.9475 | 1.0000 | 0.9730 |
| MNB + TF-IDF (unigram) | 0.9747 | 0.9546 | 1.0000 | 0.9768 |
| MNB + TF-IDF (uni+bigram) | 0.9652 | 0.9387 | 1.0000 | 0.9684 |

En iyi sonuç, en basit model olan **sayma vektörleri + Multinomial Naive Bayes**'ten geldi: 948 test mesajında 18 hata, kaçırılan spam yok.

### Dengesiz veri senaryosu (%10 spam)

Gerçek hayatta SMS'lerin çok az bir kısmı spam. Bu durumu canlandırmak için spam oranı %10'a düşürüldü.

| Model | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) |
|---|---|---|---|---|
| Her şeye "normal" diyen model | 0.9004 | 0.0000 | 0.0000 | 0.0000 |
| MNB + Count | 0.9959 | 0.9796 | 0.9796 | 0.9796 |
| Markov-Bayes (prior'lı) | 0.9878 | 1.0000 | 0.8776 | 0.9348 |
| Markov-Bayes (prior'sız) | 0.9736 | 0.8462 | 0.8980 | 0.8713 |

## Öne çıkan bulgular

* **Dengesiz veride accuracy yanıltıcı:** Hiçbir şey öğrenmeyen, her mesaja "normal" diyen model bile %90 accuracy alıyor, ama tek bir spam'i yakalamıyor (F1 = 0).
* **`<unk>` token'ını öğretmek önemli:** Sıfırdan yazılan Markov-Bayes modelinde eğitimde tek geçen kelimeler `<unk>` yapılınca F1 **0.891'den 0.970'e** çıktı. Bu yapılmadığında model, bilinmeyen kelimelerden oluşan kısa normal mesajları ("Geliom", "Annemdeyiz" gibi) spam sanıyor.
* **Hataların neredeyse hepsi kısa normal mesajlar:** En iyi modelin 18 hatasının tamamı spam sanılan kısa normal mesajlar. Bu mesajların kelimeleri sözlükte olmadığı için vektörleri sıfır oluyor ve karar sadece prior'a kalıyor.
* **Prior'ın rolü:** Dengesiz senaryoda Markov-Bayes'ten prior çıkarıldığında precision 1.00'dan 0.85'e düşüyor. Bayes teoremindeki prior, "spam nadirdir" bilgisini modele katıyor.
* **Özel token'lar** (`<url>`, `<tel>`, `<para>`, `<sayi>`) başarıyı değiştirmedi, ama sözlüğü 12.467'den 10.591 kelimeye (~%15) küçülttü.
* **Spam kampanyaları:** Kosinüs benzerliğiyle spam mesajların yaklaşık **dörtte birinin** (%26,5), aynı kampanyanın küçük farklarla (numara, tarih, fiyat) tekrar gönderilmiş kopyaları olduğu bulundu.
* **Yapay spam:** Spam mesajlarla eğitilen Markov modelinin ürettiği 1000 yapay mesajın %99,9'u sınıflandırıcı tarafından spam olarak tanındı.

## Çalıştırma

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
jupyter notebook sms_spam_tespiti.ipynb
```

`TurkishSMSCollection.csv` dosyası notebook ile aynı klasörde olmalı.

Kullanılan sürümler: Python 3.13, pandas 3.0, numpy 2.5, scikit-learn 1.9, matplotlib 3.11.

## Dosyalar

```
├── README.md
├── sms_spam_tespiti.ipynb     # tüm kod, grafikler ve sonuçlar
└── TurkishSMSCollection.csv   # veri seti (Kaggle)
```

## Sınırlılıklar ve geliştirme fikirleri

* Veri seti birkaç bin mesajlık ve belli bir döneme ait. Yeni tür spam'ler (kargo, banka dolandırıcılığı vb.) farklı bir dil kullanabilir.
* Spam mesajların çoğu Türkçe karakter kullanmıyor (ör. "DIGITURK"), bu da Türkçe küçük harf dönüşümünde bazı bozulmalara yol açıyor ("dıgıturk").
* Kök bulma (stemming), karar ağaçları ve çapraz doğrulama ile daha sağlam bir değerlendirme yapılabilir.
