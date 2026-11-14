# یافته‌های اجرایی — تشخیص آنومالی فنولوژی‌محور در مزارع نیشکر

پوشه `docs/findings` ثبت یافته‌های حاصل از اجرای واقعی pipeline روی داده‌های متریک خوزستان است.
هر فایل یک یافتهٔ مستقل را با شواهد عددی مستند می‌کند تا برای ارائه به استاد راهنما و دفاع قابل استناد باشد.

## فهرست

1. `01_data_and_ingest.md` — داده و ingest
2. `02_phenology_cycles.md` — فنولوژی و چرخه‌ها
3. `03_baseline_indices_vs_raw.md` — مقایسهٔ baseline روی شاخص‌ها در برابر باندهای خام
4. `04_evaluation.md` — ارزیابی و تزریق مصنوعی
5. `05_ssl_training.md` — آموزش SSL و پایداری
6. `06_advisor_presentation_plan.md` — چه چیزی بعد از ارائهٔ پاورپوینت به استاد نشان دهیم
7. `07_validation_and_map.md` — validation split و نقشهٔ مکانی پویا (per-day anomaly)
8. `08_honest_assessment.md` — ارزیابی صادقانه: شدنی بودن، معماری‌ها، محدودیت‌ها، نقاط قوت
9. `09_anomaly_layer_explained.md` — لایهٔ آنومالی: معنا، منشأ، فرمول‌ها، علم پشتش
10. `10_remaining_code_results.md` — نتایج تکمیلی: بک‌اند Rust (`32.7×`)، تحلیل حساسیت Ablation، آزمون درون‌مزرعه‌ای `E-INTRA`، مقایسهٔ `B0–B3` و `GRU-AE` در برابر `Masked Transformer`
11. `11_rq1_to_rq8_empirical_framework.md` — چارچوب تجربی کامل ادبیات پژوهش (`RQ1–RQ8`، شکاف‌های `G1–G8`، دروازهٔ اعتبارسنجی مستقل `Anomaly ≠ Stress`، مقاومت ابری `date/Δt`، پیش‌دستی `+34` روزه و `Temporal XAI`)
12. `12_data_scarcity_diagnostics.md` — تشخیص کمبود داده و کیفیت بازنمایی (`DS1–DS5`: learning curve، train/val gap، collapse/alignment/uniformity + t-SNE، few-shot، واریانس seed/LOYO) + پروتکل_eval مدل‌های آینده در `docs/eval_protocol_future_models.md`
13. `13_literature_phenology_and_perturbations.md` — مرور منابع: استخراج خودکار فنولوژی/چرخه، محور زمان برای SSL، اختلال‌های نیشکر و SSL روی سری نامنظم (۳۹ منبع، ۲۲ موجود + ۱۷ تازه؛ پیشنهادهای R1–R8 برای سند تصمیم)
14. `14_cycle_audit.md` — ممیزی segmentation: نمونهٔ ۹۴ چرخه‌ای، برچسب‌های عامل (۶۷ درست / ۱۹ merged / ۶ split / ۲ جابه‌جایی)، نرخ خطای جمعیت ۷.۹٪ و قاعدهٔ حذف ۳۸ چرخه
15. `15_cycle_segmentation_method.md` — شرح کامل روش استخراج چرخه‌ها (گام‌به‌گام با پارامترها و file:line) و تطبیق با منابع مرورشده (Kavats، Joshi، Manrique، Rodigheri، BFAST، CCDC، GDD) + خطای شناخته‌شدهٔ merged زمستانی و قاعدهٔ حذف
16. `16_development_protocol_results.md` — نتایج پروتکل توسعه: معماری نهایی (`gru_ae_main`)، آمار هشدار، یافته‌ی شدت ۱×MAD، تست نهایی قفل‌شده (پله L2؛ هدف D6 برآورده نشد و صادقانه ثبت شد)
17. `17_stage_conditioned_thresholds.md` — آستانه‌های شرطی-مرحله‌ای با پوش اطمینان npphen (جلوگیری از پرچ‌کردن افت فنولوژیک طبیعی) روی ۷۹۹ چرخه و ۳۷۰ مزرعه: ۱۸,۵۰۵ مشاهده، ۱,۷۷۴ شانه (۹.۵۹%). **تغییر شماره از ۱۴ به ۱۷ (2026-11-14): دو یافته شمارهٔ 14 داشتند؛ شمارهٔ ۱۴ به `14_cycle_audit.md` رسید.**
