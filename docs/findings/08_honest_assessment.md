# یافته ۸ — ارزیابی صادقانه: شدنی بودن، محدودیت‌ها، نقاط قوت

## ۱. آیا پروپوزال شدنی است؟

**بله، اما با شرایط.** پروپوزال از نظر علمی قابل اجراست، اما باید انتظارات را مدیریت کنیم:

- **فاز ۰ و ۱ (ingest + baseline)**: کاملاً شدنی و انجام شد. baseline آماری phase-aware روی دادهٔ واقعی کار می‌کند.
- **فاز ۲ (SSL)**: کد اجرا می‌شود، اما روی ۳۲ مزرعه (۴۳ cycle) کوچک است. برای ادعای Q1 باید روی ۴۹۰ مزرعه اجرا شود.
- **Gate G2 (SSL beats B1 with CI)**: هنوز تست نشده. این gate تعیین‌کننده است.

## ۲. معماری‌هایی که باید تست شوند (به ترتیب)

| کد | مدل | سخت‌افزار | نقش | وضعیت |
|---|---|---|---|---|
| B0 | z-score تقویمی (DOY) | CPU | کف مقایسه | باید اجرا شود |
| B1 | z-score phase-aware (dsh) | CPU | هستهٔ پروژه | ✅ انجام شد |
| B2 | B1 با محور GDD/landmark | CPU | انتخاب محور | باید اجرا شود |
| B3 | IsolationForest/ECOD | CPU | baseline غیرپارامتری | باید اجرا شود |
| M-A | GRU-AE | CPU | baseline یادگیری عمیق | کد موجود، اجرا نشده |
| M-B | Masked Transformer (161K) | CPU/GPU | مدل اصلی SSL | ✅ انجام شد |
| M-C | CNN-Mamba | GPU/Colab | مقایسه | کد موجود نیست |
| M-D | 3D-CAE | GPU | فاز ۳ (patch) | کد موجود نیست |

## ۳. توقعات واقع‌بینانه

- **Gate G1 (B1 beats B0)**: احتمال بالا. phase-aware باید variance را کاهش دهد.
- **Gate G2 (SSL beats B1)**: **نامشخص.** با ۴۳ cycle، SSL ممکن است overfit شود. باید روی ۴۹۰ مزرعه تست شود.
- **اگر SSL نبرد**: این خود یک یافتهٔ مهم است (G2) و باید گزارش شود. SSL به embedding/retrieval محدود می‌شود.
- **Recall فعلی (۰.۰۲)**: پایین است. baseline آماری به‌تنهایی حساسیت کمی دارد. این توجیه ورود SSL است.

## ۴. محدودیت‌های اصلی

1. **بدون برچسب (label-free)**: هیچ ground truth وجود ندارد. اعتبارسنجی با proxyها (CWSI, ETa) است که نویزی‌اند.
2. **CWSI ↔ LST دایره‌ای**: CWSI از LST ساخته می‌شود. استفاده از CWSI به‌عنوان proxy، نتیجهٔ خوش‌بینانه می‌دهد. باید ablation بدون LST انجام شود.
3. **تعداد کم cycle**: ۴۳ cycle برای SSL کافی نیست. باید روی ۴۹۰ مزرعه (هزاران cycle) اجرا شود.
4. **Landsat 16-روزه**: cadence نامنظم، ۶۸ صحنهٔ حذف‌شده (ابر/خطا).
5. **بدون patch-level**: فقط field-mean. ناهمگونی درون‌مزرعه‌ای نادیده گرفته می‌شود.
6. **Rust نیست**: gate G3. کد Python ممکن است روی ۴۹۰ مزرعه کند باشد.
7. **GPU نداریم (Windows)**: SSL روی CPU. CNN-Mamba نیاز به GPU دارد.

## ۵. نقاط قوت نسبت به ادبیات (thesis_research/SSL_Proposal)

| شکاف در ادبیات | پاسخ این پروپوزال |
|---|---|
| STS-Net: فقط NDVI، فقط Planet 3m، supervised | ما: multi-band (raw+LST) + Landsat 30m + label-free |
| SAED-ViT: MAE-ViT سنگین (۱۵M params)، خوشه‌بندی | ما: ۱۶۱K params (frugal)، بدون خوشه‌بندی |
| EigenCL: فقط NDRE تک‌بعدی، غفلت از حرارتی | ما: کل طیف انعکاسی-حرارتی |
| Time2Agri: پیش‌آموزش فنولوژیک اما anomaly نه | ما: anomaly detection + phenology alignment |
| Presto: طبقه‌بندی، نه anomaly | ما: anomaly score بومی (reconstruction error) |
| مقالات کلی: صرفاً شاخص‌های پوشش | ما: raw bands + LST + weather (GDD) |
| عدم مقایسه بین‌ساله در فاز فنولوژیک همسان | ما: days-since-harvest + stratum fallback |
| مدل‌های غول‌آسا | ما: frugal (۱۶۱K params، روی لپ‌تاپ) |

## ۶. توصیهٔ نهایی

1. **ابتدا B0/B1/B2/B3 را روی ۴۹۰ مزرعه اجرا کنید** (gate G1).
2. **سپس SSL را روی ۴۹۰ مزرعه اجرا کنید** (gate G2).
3. **ablation بدون LST** را انجام دهید (پاسخ به circularity).
4. **E-INTRA** را روی ۳۲ مزرعهٔ subset اجرا کنید (آیا field-mean کفایت می‌کند؟).
5. **اگر SSL نبرد**: گزارش کنید که SSL به embedding/retrieval محدود می‌شود (این خود یافته است).

## ۷. چه چیزی را به استاد بگویید

- «پروپوزال شدنی است، اما gate G2 (SSL beats B1) باید روی ۴۹۰ مزرعه تست شود.»
- «نقاط قوت: frugal (۱۶۱K params)، label-free، multi-band + weather، phase-aware.»
- «محدودیت‌ها: بدون برچسب، CWSI circular، ۴۳ cycle برای SSL کم است.»
- «Next step: اجرای کامل روی ۴۹۰ مزرعه + ablation + E-INTRA.»