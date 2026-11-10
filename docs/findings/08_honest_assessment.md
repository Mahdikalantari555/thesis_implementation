# یافته ۸ — ارزیابی صادقانه: شدنی بودن، محدودیت‌ها، نقاط قوت

## ۱. آیا پروپوزال شدنی است؟

**بله، اما با شرایط.** پروپوزال از نظر علمی قابل اجراست، اما باید انتظارات را مدیریت کنیم:

- **فاز ۰ و ۱ (ingest + baseline)**: کاملاً شدنی و انجام شد. baseline آماری phase-aware روی دادهٔ واقعی کار می‌کند.
- **فاز ۲ (SSL)**: کد اجرا می‌شود، اما روی ۳۲ مزرعه (۴۳ cycle) کوچک است. برای ادعای Q1 باید روی ۴۹۰ مزرعه اجرا شود.
- **Gate G2 (SSL beats B1 with CI)**: هنوز تست نشده. این gate تعیین‌کننده است.

## ۲. معماری‌هایی که باید تست شوند (به ترتیب)

| کد | مدل | سخت‌افزار | نقش | وضعیت |
|---|---|---|---|---|
| B0 | z-score تقویمی (DOY) | CPU | کف مقایسه | ✅ انجام شد (AUC-PR = 0.885) |
| B1 | z-score phase-aware (dsh) | CPU | هستهٔ پروژه | ✅ انجام شد (AUC-PR = 0.893) |
| B2 | B1 با محور GDD / landmark | CPU | انتخاب محور | ✅ انجام شد (B2-GDD بهترین: AUC-PR = 0.901, ρ_CWSI = 0.649) |
| B3 | IsolationForest / ECOD | CPU | baseline غیرپارامتری | ✅ انجام شد (ECOD = 0.747, IF = 0.629) |
| M-A | GRU-AE (161K) | CPU | baseline یادگیری عمیق | ✅ انجام شد (recon AUC-PR = 0.854) |
| M-B | Masked Transformer (161K) | CPU/GPU | مدل اصلی SSL | ✅ انجام شد (recon AUC-PR = 0.902) |
| Rust | `ssl_anomaly_rs` (PyO3 + Rayon) | CPU | شتاب‌دهندهٔ هسته | ⚠️ کامپایل شد؛ revendication شتاب ۳۲.۷× در Whittaker **بازتولید نشده** (بازبینی در دستorescence — گزینه‌های preserve/remove در todo.md) |
| M-C | CNN-Mamba | GPU/Colab | مقایسه | خارج از دامنهٔ فعلی (نیازمند GPU) |
| M-D | 3D-CAE | GPU | فاز ۳ (patch) | موکول به فاز ۳ (E-INTRA: میانگین پراکندگی ۹.۳ روز < ۳۰ روز) |

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
8. **آرشیو واقعی ۴۹۰ مزرعه بدون برچسب تنش:** `load_real490_archive()` آرشیو واقعی ۴۹۰ مزرعه را بارگذاری می‌کند (799 cycle غیرسانسور، ۳۷۰ مزرعه، ۵۳۹ تاریخ) اما **برچسب تنش Verify شده (y_stress) غیرفعال است** تا زمان اتصال prodact‌های zonal METRIC (CWSI_ET / ETaDaily). در این interim، `y_stress_proxy = weak_proxy_top20pct_mean_cwsi_not_validated_stress` و `eta_deficit/cwsi = absent` گزارش می‌شوند. هر گزارش اعتبارسنجی بر اساس آرشیو ۴۹۰،恥اند باید این غیاب را صراحتاً بیان کند.

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
6. **چonanMetrics ۴۹۰**: زonal stats METRIC (CWSI_ET / ETaDaily) را به صورت per-field-per-date wire کنید تا برچسب تنش verify شود (task `visualize-metric-products`).

## ۷. چه چیزی را به استاد بگویید (به‌روزرسانی نهایی پس از اجرای `RQ1–RQ8`)

- **جایگاه دقیق علمی (بدون ادعای اغراق‌آمیز)**: ما ادعای «اولین تشخیص تنش بدون برچسب» را نداریم (*EigenCL* و *Sadbhave* پیش‌تر بوده‌اند)؛ نوآوری دقیق ما **«اولین چارچوب خودنظارتی آگاه از فاز روی آرشیو بلندمدت Landsat در سطح مزرعه با ورودی هم‌زمان نوری + حرارتی (`LST`)»** است.
- **مرز سخت‌گیرانهٔ `Anomaly ≠ Stress`**: خروجی خام مدل‌ها تا پیش از عبور از دروازهٔ چندپروکسی غیردایره‌ای (`validation_gate.py`: کمبود `ETa` + مدل بدون `LST` در برابر `CWSI` + دقت `Top-K`) صرفاً «آنومالی طیفی-حرارتی» نامیده می‌شود، نه «تنش».
- **پاسخ تجربی کامل به `RQ1–RQ8`**:
  - `RQ1`: در بودجهٔ هشدار ۱۰٪ تحت `LOYO`، مدل `SSL` (`Recall@10% = 0.718`) ۳ برابر بهتر از آستانهٔ `NDVI` (`0.241`) و `+11.8` تا `+14.1` واحد درصد بهتر از `GBT`/`RF` عمل می‌کند. *(منبع: آرشیو شبیه‌سازی‌شدهٔ ۴۹۰ مزرعه؛ real490 بدون برچسب تنش Verify شده lacks)*
  - `RQ2`: پیش‌آموزش `anomaly_degradation` بالاترین دقت موضعی (`AUC-PR = 0.990`) و `traj_contrastive` بالاترین همبستگی تنش کل فصل (`CWSI` $\rho = +0.433$) را دارد. *(منبع: آرشیو شبیه‌سازی‌شدهٔ ۴۹۰ مزرعه)*
  - `RQ3`: افزودن غیردایره‌ای `LST` در سطح `SSL` تشخیص تنش روزنه‌ای-پوششی را `+0.249` واحد `AUC-PR` ارتقا داده و $R^2$ پروب منجمد `ETa` را ۳ برابر (`0.068 -> 0.203`) می‌کند. *(منبع: آرشیو شبیه‌سازی‌شدهٔ ۴۹۰ مزرعه)*
  - `RQ4 & RQ5`: شرطی‌سازی صریح `date/Δt` در پوشش ۵۰٪ ابری ۸۵.۴٪ کارایی را حفظ می‌کند و مدل علّی چندبانده به طور میانگین **`+34` روز زودتر** از آستانهٔ `NDVI` هشدار می‌دهد. *(منبع: آرشیو شبیه‌سازی‌شدهٔ ۴۹۰ مزرعه)*
  - `RQ6–RQ8`: انتساب `Integrated Gradients` با **`95.6%` هم‌خوانی فیزیولوژی گیاهی** وفادارترین تبیین زمانی را ارائه می‌دهد و پیرایش `MAD` حتی در ۳۰٪ آلودگی دادهٔ آموزش، `AUC-PR = 0.877` را حفظ می‌کند. *(منبع: آرشیو شبیه‌سازی‌شدهٔ ۴۹۰ مزرعه)*
- **گام عملیاتی پس از اتصال دیسک محلی ۴۹۰ مزرعه**: اجرای مستقیم `ssl-anomaly research --rq all` روی آرشیو کامل محلی و تکمیل ستون ممیزی کارشناس زراعت در `reports/expert_stress_review_packet.csv`.
- **چonan نحوهٔ اجرای بنچمارک با دادهٔ واقعی ۴۹۰:**
  - `python scripts/run_rq1_benchmark.py --source real490` → `reports/rq1_benchmark.json` (data_source="real490", fail-closed اگر فایل‌ها گم باشند).
  - `python scripts/run_rq2_pretexts.py --source real490` → `reports/rq2_pretexts.json`
  - `python scripts/run_rq3_lst_ssl.py --source real490` → `reports/rq3_lst_ssl.json`
  - `python scripts/run_rq4_cloud_and_stability.py --source real490` → `reports/rq4_cloud_and_stability.json`
  - `python scripts/run_rq5_early_detection.py --source real490 --n-eval 20` → `reports/rq5_early_detection.json`
  - `python scripts/run_rq6_xai.py --source real490 --n-eval 20` → `reports/rq6_xai.json`
  - `python scripts/run_rq7_rq8.py --source real490` → `reports/rq7_rq8.json`
  - `python scripts/run_data_scarcity.py --source real490 --arms anomaly_degradation --seeds 42 --epochs 10` → `reports/data_scarcity.json`
  - برای خروجی شبیه‌ساز: `--source simulated` → فایل‌ها زیر `reports/simulated/` نوشته می‌شوند و `data_source="simulated"` ثبت می‌شود.
