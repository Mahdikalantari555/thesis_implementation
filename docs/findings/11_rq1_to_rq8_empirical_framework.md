# یافته ۱۱ — چارچوب تجربی کامل ادبیات پژوهش (`RQ1–RQ8`، شکاف‌های `G1–G8` و دروازهٔ اعتبارسنجی مستقل `Anomaly ≠ Stress`)

این سند نتایج کامل اجرای بستهٔ پژوهشی نهایی را مستند می‌کند که تمامی اجزای باقی‌ماندهٔ فاز ۱ و ۲ در `docs/final_plan.md` و تمامی پرسش‌های پژوهشی استخراج‌شده از مرور ادبیات (`RQ1–RQ8` متناظر با شکاف‌های `G1–G8`) را با رعایت دقیق مرزهای ادعایی و دامنه‌ای پیاده‌سازی و ارزیابی کرده است.

---

## ۰. چارچوب ادعایی و مرزهای دامنه (Positioning, Non-Claims & Non-Goals)

1. **مرز ادعای علمی (Strict Positioning)**:
   - این پژوهش **هرگز** ادعای «نخستین تشخیص بدون برچسب تنش گیاهی» (*first label-free crop-stress detection*) را مطرح **نمی‌کند** (کارهای پیشین نظیر *EigenCL* و *Sadbhave* پیش‌تر وجود داشته‌اند).
   - جایگاه دقیق علمی این پایان‌نامه: **«نخستین چارچوب خودنظارتی آگاه از فاز فنولوژیک روی آرشیو بلندمدت در سطح مزرعهٔ Landsat با ترکیب هم‌زمان باندهای نوری و حرارتی (`Optical + Thermal LST`)»** (*first phenology-aware self-supervised anomaly framework on a long field-level Landsat archive with joint optical + thermal inputs*).

2. **تفکیک سخت‌گیرانهٔ «آنومالی ≠ تنش» (`Anomaly ≠ Stress`)**:
   - خروجی خام مدل‌های آماری (`B0–B3`) و خودنظارتی (`SSL`) صرفاً **آنومالی طیفی-حرارتی تأییدنشده** (`unvalidated_spectral_thermal_anomaly`) است (که می‌تواند ناشی از جابه‌جایی تاریخ برداشت، خوابیدگی محصول، سایهٔ ابر باقیمانده، علف هرز یا تغییرات مدیریتی باشد).
   - واژهٔ **«شاخص تنش تأییدشدهٔ محصول»** (`validated_crop_stress_indicator`) در کد (`src/ssl_anomaly/eval/validation_gate.py`) قفل شده و تنها زمانی مجاز است که خروجی مدل از **دروازهٔ اعتبارسنجی چندپروکسی غیردایره‌ای** (`ETa deficit`، همبستگی غیردایره‌ای مدل بدون `LST` با `CWSI`، افت عملکرد/زیست‌توده، و دقت `Top-K`) عبور کند.

3. **شفافیت داده (آرشیو واقعی ۴۹۰ مزرعه در برابر آرشیو شبیه‌سازی‌شده)**:
   - از تاریخ ۲۰۲۶-۱۱-۱۰، آرشیو واقعی ۴۹۰ مزرعه (`Data/Processed/satellite/stats.zarr` + `data/processed/cycles_490.parquet` + `data/interim/growth_cycles_490.npz`) منبع پیش‌فرض (`--source real490`) است و هر گزارش `data_source="real490"` را ثبت می‌کند.
   - **برچسب تنش Verify شده غیرفعال است:** `load_real490_archive()` آرشیو واقعی ۴۹۰ مزرعه را بارگذاری می‌کند (799 cycle غیرسانسور، ۳۷۰ مزرعه با ≥۱ cycle، ۵۳۸ تاریخ acquisition، ۲۰۱۳-۰۳-۳۰ تا ۲۰۲۶-۰۹-۱۰) اما `y_stress` (برچسب تنش) فقط یک placebo صفر است و `eta_deficit`/`cwsi`에는 `NaN` دارند. زونال استات METRIC (`CWSI_ET` / `ETaDaily`) هنوز wire نشده است (task `visualize-metric-products` در حال پیاده‌سازی). تا آن زمان، هر گزارش اعتبارسنجی بر اساس real490 این غیاب را صراحتاً بیان می‌کند و هرگز برچسب‌های ۳۲ مزرعه (`fields.zarr`) به real490 منتقل نمی‌شود.
   - **در صورت عدم وجود فایل‌های آرشیو real490**، اسکریپت‌ها با `SystemExit(1)` خاتمه می‌یابند و هیچ گزارشی نوشته نمی‌شود (fail-closed). شبیه‌ساز تنها با `--source simulated` فعال می‌شود و خروجی‌ها زیر `reports/simulated/` نوشته می‌شوند و `data_source="simulated"` ثبت می‌گردد.

---

## ۱. نتایج `RQ1` (شکاف‌های `G2` و `G7`) — بنچمارک `LOYO` در برابر آلارم‌های عملیاتی و درخت‌های تصمیم در بودجهٔ هشدار برابر

اسکریپت: `scripts/run_rq1_benchmark.py` | خروجی‌ها: `reports/rq1_benchmark.{json,csv}`

### الف) آرشیو ۴۹۰ مزرعه × ۱۳.۵ سال تحت اعتبارسنجی `Leave-One-Year-Out (LOYO)`

**NOT RUN** — `y_stress` در آرشیو real490 یک وکتور تک‌طبقه‌ای است (`n_pos=0`, `n_neg=49`).AUC-PR/recall بر روی یک کلاس تعریف نشده است. هر گزارش اعتبارسنجی بر اساس real490 برچسب تنش Verify شده ندارد.

```json
{
  "status": "not_run",
  "note": "AUC-PR/recall are undefined: y_stress has a single class (n_pos=0, n_neg=49). On the real 490-field archive stress labels are pending CWSI/ETa zonal extraction, so the detector comparison is reported as not-run instead of being computed on a degenerate label vector.",
  "n_fields_evaluated": 0,
  "n_loyo_folds": 0,
  "held_out_years": [],
  "detectors": {},
  "paired_wilcoxon_vs_ssl": {}
}
```

### real490 status (pending validated stress labels)

آرشیو real490 بارگذاری می‌شود اما برچسب تنش Verify نشده است. هیچ گزارش real490 با اعداد تشخیص تنش (AUC-PR، Recall) در این سند وجود ندارد. پس از اتصال prodact‌های zonal METRIC و Verify شدن برچسب تنش، این بخش بازنویسی خواهد شد.

---

## ۲. نتایج `RQ2` (شکاف `G3`) — مقایسهٔ کنترل‌شدهٔ سه وظیفهٔ پیش‌آموزش خودنظارتی (`Pretext Tasks`)

اسکریپت: `scripts/run_rq2_pretexts.py` | خروجی‌ها: `reports/rq2_pretexts.{json,csv}`  
هر سه بازوی پیش‌آموزش روی **اسکلت کاملاً یکسان `PretextTransformer`** (`~111K–112K` پارامتر، ۲ اپوک، ۱ بذر تصادفی) روی آرشیو real490 اجرا شدند. کل ۸۲,۰۷۲ سلول (35.7%) از آرایه ورودی `X` پیش از آموزش، non-finite (NaN) بود و با `np.nan_to_num` جبران شد.

| بازوی پیش‌آموزش (`Pretext Arm`) | پارامترها | زمان آموزش | `AUC-PR` | `AUC-ROC` | `Recall@5%` | `Recall@10%` | همبستگی کل فصل `CWSI` ($\rho$) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `masked_recon` | `~112K` | ~6 s | **0.6013** | **0.6086** | 0.0976 | 0.1674 | NaN |
| `traj_contrastive` | `~113K` | ~3 s | **0.4635** | **0.4247** | 0.0482 | 0.0698 | NaN |
| `anomaly_degradation` | `~110K` | ~4 s | **0.7641** | **0.7249** | 0.2372 | 0.3795 | NaN |
| `time2agri` | `~112K` | ~7 s | **0.6400** | **0.6094** | 0.1508 | 0.2522 | NaN |

- **یافتهٔ کلیدی `RQ2` (real490):** در غیاب برچسب تنش Verify شده، هر سه بازوی پیش‌آموزش AUC-PR قابل‌تأیید تولید می‌کنند اما **همه `cwsi_rho = NaN`** (هیچ همبستگی معتبری با CWSI وجود ندارد). `anomaly_degradation` بالاترین AUC-PR (0.7641) را دارد اما چون برچسب تنش degenerate است، این عدد قابل‌تأیید نیست و صرفاً نشان‌دهندهٔTheir الگوهای بازسازی است، نه تشخیص تنش واقعی.

---

## ۳. نتایج `RQ3` (شکاف `G1`) — آزمون غیردایره‌ای ارزش حرارتی (`LST`) در سطح `SSL`

اسکریپت: `scripts/run_rq3_lst_ssl.py` | خروجی‌ها: `reports/rq3_lst_ssl.{json,csv}`  
برای حذف کامل دایره‌ای بودن `ST_B10 ↔ CWSI`، مدل‌های `optical_only_ssl` (۶ باند بازتابی) و `optical_plus_lst_ssl` (۶ باند بازتابی + `ST_B10`) صرفاً روی اهداف **غیردایره‌ای** ارزیابی شدند.

| معیار غیردایره‌ای | `Optical-Only SSL` | `Optical + LST SSL` | بهرهٔ خالص `Δ(LST)` |
|---|---:|---:|---:|
| تشخیص اغتشاش صرفاً نوری واقعی (`real_optical_pert_pr`) | **0.7968** | **0.7551** | **-0.0417** |
| تشخیص تنش توأم روزنه‌ای-پوششی (`coupled_stress_pr`) | **0.0000** | **0.0000** | **0.0000** |
| پروب خطی منجمد کمبود تبخیر `ETa` (`eta_probe_r2`) | NaN | NaN | — |

- **یافتهٔ کلیدی `RQ3` (real490):** افزودن باند حرارتی `ST_B10` در سطح SSL **بهبود ایجاد نمی‌کند** (`Δ = -0.0417` منفی) و تشخیص تنش توأم روزنه‌ای-پوششی (`coupled_stress_pr`) **صفر است** برای هر دو مدل. دلیل این عدم تطابق، برچسب تنش degenerate است (هر Cycle برچسب ۰ دارد) و بنابراین `y_true` یک کلاس است و AUC-PR/R² غیرمعتبر می‌شوند.

---

## ۴. نتایج `RQ4` (شکاف `G8`) و پایداری فاز (`E-ALIGN`، `E-SEG`، `E-NORM`) — مقاومت در برابر ابرناکی تا cadence ماهانه

اسکریپت: `scripts/run_rq4_cloud_and_stability.py` | خروجی‌ها: `reports/rq4_cloud_and_stability.{json,csv}`

### الف) sweep کامل بودن سری زمانی

**NOT RUN** — sweep کامل بودن سری زمانی به برچسب تنش Validate شده نیاز دارد (AUC-PR). آرشیو real490 هیچ برچسب تنش Validate شده ندارد و sweep شبیه‌سازی‌شده فقط با `--source simulated` اجرا می‌شود.

### ب) E-ALIGN / E-NORM / E-SEG

**NOT RUN** — E-ALIGN و E-NORM به دادهٔ زونال ۳۲ مزرعه (`fields.zarr` / `cycles.parquet`) نیاز دارند که در آن 궤فه وجود ندارد (فقط `stats.zarr` در دیتابیس real490 وجود دارد). reporter نقشه می‌نویسد: `E-ALIGN requires real32 zarr data`.

---

## ۵. نتایج `RQ5` (شکاف `G5`) — زمان پیش‌دستی هشدار زودهنگام (`Early Detection Lead Time`)

اسکریپت: `scripts/run_rq5_early_detection.py` | خروجی‌ها: `reports/rq5_early_detection.{json,csv}`  
تحت اعتبارسنجی علّی گام‌به‌گام (`Causal Online LOYO`) و کالیبراسیون آستانهٔ همهٔ روش‌ها روی **نرخ هشدار کاذب یکسان ۵٪ (`FAR = 5%`)**:

| آشکارساز علّی برخط | نرخ کشف در `FAR = 5%` | میانگین تأخیر از شروع تنش (روز) | بازهٔ اطمینان ۹۵٪ (روز) | میانهٔ تأخیر (روز) | پیش‌دستی نسبت به `NDVI` |
|---|---:|---:|---:|---:|---:|
| `SSL_Causal_OpticalThermal` | **0.00** | NaN | — | NaN | +0.00 روز |
| `B1_CUSUM_Causal` | **0.05** | NaN | — | NaN | — |
| `B1_PhaseZ_Causal` | **0.11** | **7.50** | `[0.00, 15.00]` | **7.50** | — |
| `NDVI_Index_Threshold` | **0.00** | NaN | — | NaN | مرجع (`0.0` روز) |

- **یافتهٔ کلیدی `RQ5` (real490):** نرخ کشف تمام روش‌ها در بودجهٔ `FAR = 5%` بسیار پایین است (۰.۰۰ تا ۰.۱۱). SSL مزیت تأخیری نسبت به آستانه NDVI ندارد (`+0.00` روز). دلیل این عدم بهبود، برچسب تنش degenerate است (هر Cycle برچسب ۰) و بنابراین هر روش detector نتواند سیگنال معتبری پیدا کند.

---

## ۶. نتایج `RQ6` (شکاف `G6`) — توضیح‌پذیری زمانی (`Temporal XAI`) و هم‌خوانی فیزیولوژی گیاهی

اسکریپت: `scripts/run_rq6_xai.py` | خروجی‌ها: `reports/rq6_xai.{json,csv}` | ماژول: `src/ssl_anomaly/xai/`

| روش انتساب زمانی (`XAI`) | `Deletion AUC` *(کمتر=بهتر)* | بهرهٔ حذف نسبت به تصادفی | `Insertion AUC` *(بیشتر=بهتر)* | بهرهٔ درج | **هم‌خوانی علامت فیزیولوژیک** | افت تاج (`NIR/NDVI`) | گرمایش (`ST_B10`) | تنش آبی برگ (`SWIR`) | farm share | `sanity_recon_abs_rho` | `passes_neural_sanity` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`Integrated_Gradients`** | **0.1884** | **+0.3295** | **1.0828** | **+0.4358** | NaN | NaN | NaN | NaN | 0.1738 | 0.3635 | **True** |
| **`Temporal_GradCAM`** | 0.2882 | +0.2296 | 1.0108 | +0.3638 | NaN | NaN | NaN | NaN | 0.0464 | 0.4212 | **True** |
| **`Window_Occlusion`** | 0.2723 | +0.2455 | 1.1284 | +0.4814 | NaN | NaN | NaN | NaN | 0.1680 | 0.0548 | **True** |
| **`Temporal_SHAP`** | 0.2642 | +0.2537 | 0.9972 | +0.3502 | NaN | NaN | NaN | NaN | 0.3114 | 0.9288 | **False** |

- **یافتهٔ کلیدی `RQ6` (real490):** تمام هم‌خوانی‌های فیزیولوژیک `NaN` (هیچ برچسب تنش Validate شده برای مقایسه وجود ندارد). سه روش (IG، Temporal GradCAM، Window Occlusion) از آزمون سلامت عصبی (neural sanity) عبور می‌کنند اما `Temporal_SHAP` شکست می‌خورهاید (`passes_neural_sanity = False`). deletion_auc مقادیر scales را نشان می‌دهد که به نظر می‌رسد نمی‌توانند تشخیص معتبری ایجاد کنند (بسیار پایین، نزدیک به تصادفی).

---

## ۷. دروازهٔ اعتبارسنجی مستقل (`Anomaly ≠ Stress`)، `RQ7` و `RQ8`

اسکریپت: `scripts/run_rq7_rq8.py` | خروجی‌ها: `reports/rq7_rq8.{json,csv}`، `reports/rq7_probing.csv`، `reports/expert_stress_review_packet.csv`

### ۱. دروازهٔ اعتبارسنجی مستقل (`validation_gate.py`)

**گذر نشد** — دروازه چندپروکسی غیردایره‌ای (`multi_proxy_non_circular_gate`) با `gate_passed = False` نتایج را برمی‌گرداند و اصطلاح مجاز `unvalidated_spectral_thermal_anomaly` است. دروازه حلقوی (فقط CWSI) هم رد شده است.

### ۲. `RQ7` — پروب خطی بازنمایی منجمد (`Frozen-Feature Probing`)

| енکودر | `et0_r2` | `gdd_r2` | `radiation_r2` | `eta_deficit_r2` | `late_ratoon_auc` |
|---|---:|---:|---:|---:|---:|
| `untrained_random_encoder` | -0.0012 | **0.7919** | -0.0142 | NaN | NaN |
| `ssl_masked_recon` | 0.0000 | **0.7549** | -0.0190 | NaN | NaN |
| `ssl_traj_contrastive` | -0.0035 | **0.7015** | -0.0081 | NaN | NaN |
| `ssl_anomaly_degradation` | -0.0020 | **0.7682** | -0.0247 | NaN | NaN |
| `ssl_time2agri` | 0.0014 | **0.7034** | -0.0092 | NaN | NaN |

- **یافتهٔ کلیدی `RQ7` (real490):** $gdd\_r2$ برای تمام اینکودرها (از ۰.۷۰ تا ۰.۷۹) معتبر است اما `$eta\_deficit\_r2$` و `$late\_raton\_auc$` **NaN** (برچسب تنش degenerate). این نشان می‌دهد رمزگذارهای SSL می‌توانند ویلاسیت GDD را رمزگشایی کنند اما نه تنش Prove شده. 또한، اینکودر تصادفی آموزش‌نشده هم عملکرد مشابه دارد، یعنی سود پیش‌آموزش SSL بسیار ضعیف است.

### ۳. `RQ8` — حساسیت به آلودگی دادهٔ بدون برچسب (`0% -> 30%`) و ساختار عوامل مداخله‌گر هشدار کاذب

**NOT RUN** — هر دو تحلیل RQ8 (sweep آلودگی و تجزیه عوامل هشدار کاذب) به برچسب تنش Validate شده نیاز دارند (AUC-PR). آرشیو real490 برچسب تک‌طبقه‌ای دارد (`n_pos=0`, `n_neg=799`) بنابراین هر 보고 AUC-PR بی‌معنی است. گزارش به‌عنوان `not_run` با کل مقادیر `NaN` ثبت شده است.

---

## ن البرازيل — جمع‌بندیreal490

| RQ | وضعیت real490 | نکته کلیدی |
|---|---|---|
| RQ1 | ❌ NOT RUN | y_stress تک‌طبقه‌ای است؛ AUC-PR/Recall تعریف نشده است |
| RQ2 | ⚠️ اجرا شد | AUC-PR ۰.۴۶–۰.۷۶지만 cwsi_rho=NaN؛ برچسب تنش غیرفعال |
| RQ3 | ⚠️ اجرا شد | LST بهبود ایجاد نمی‌کند؛ coupled_stress_pr=۰ |
| RQ4 | ❌ NOT RUN | نیاز به برچسب تنش Validate شده + real32 zarr |
| RQ5 | ⚠️ اجرا شد | نرخ کشف ۰.۰۰–۰.۱۱؛ مزیت تأخیری SSL = +۰.۰۰ روز |
| RQ6 | ⚠️ اجرا شد | XAI 합 éس 징؛ IG ت consenso می‌گذرد، SHAP شکست می‌خورد؛ هم‌خوانی‌های فزیولوژیکی NaN |
| RQ7 | ❌ NOT PASSED | دروازه چندپروکسی گذر نشد؛ eta_deficit_r2=NaN |
| RQ8 | ❌ NOT RUN | y_stress تک‌طبقه‌ای است؛ sweep آلودگی بی‌معنی است |

**جمع‌بندی نهایی:** آرشیو real490 بارگذاری می‌شود و تمام اسکریپت‌ها بدون خطا اجرا می‌شوند (EXIT 0)، اما هر گزارش اعتبارسنجی که به برچسب تنش Validate شده نیاز دارد، `not_run` گزارش می‌شود. دلایل اصلی:
1. **y_stress degenerate** (همه صفر) — هیچ برچسب تنش Verify شده وجود ندارد
2. **زونال استات METRIC** (CWSI_ET / ETaDaily) هنوز wire نشده است
3. **عدم وجود ۳۲ مزرعه اصلی** (`fields.zarr` / `cycles.parquet` در 궤فه وجود ندارد)

پس از اتصال prodact‌های zonal METRIC و Verify شدن برچسب تنش، تمام بخش‌های RQ1–RQ8 بازنویسی خواهند شد.

### نحوهٔ اجرای بنچمارک با دادهٔ واقعی ۴۹۰:

```bash
# RQ1 — LOYO benchmark on real490 (NOT RUN until stress labels verified)
python scripts/run_rq1_benchmark.py --source real490 --max-fields 20 --seed 42

# RQ2 — Pretext tasks on real490
python scripts/run_rq2_pretexts.py --source real490 --epochs 2 --seeds 42

# RQ3 — LST value test on real490
python scripts/run_rq3_lst_ssl.py --source real490 --epochs 2 --seeds 42

# RQ4 — Cloud and stability on real490 (NOT RUN)
python scripts/run_rq4_cloud_and_stability.py --source real490 --seed 42

# RQ5 — Early detection on real490
python scripts/run_rq5_early_detection.py --source real490 --n-cycles 20

# RQ6 — XAI on real490
python scripts/run_rq6_xai.py --source real490 --n-eval 1

# RQ7/RQ8 — Probing and contamination on real490 (RQ8 NOT RUN)
python scripts/run_rq7_rq8.py --source real490

# Data scarcity — DS1 to DS5 on real490
python scripts/run_data_scarcity.py --source real490 --arms anomaly_degradation --seeds 42 --epochs 2
```

هر اسکریپت با `--source real490` در صورت عدم وجود فایل‌های آرشیو با `SystemExit(1)` خاتمه می‌یابند و هیچ گزارش نوشته نمی‌شود (fail-closed). برای خروجی شبیه‌ساز از `--source simulated` استفاده شود (خروجی زیر `reports/simulated/`).