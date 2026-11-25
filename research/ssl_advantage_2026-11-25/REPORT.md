# When is an SSL-advantage claim defensible in small-n field-level EO anomaly detection?

> Generated 2026-11-25 · depth: standard · 21 sources · workspace: `research/ssl_advantage_2026-11-25/`

## Executive summary

- No peer-reviewed study found here quantifies an SSL-pretraining advantage for **remote-sensing time-series anomaly detection** against a random-initialized encoder of the same architecture; the dedicated survey scopes itself to methods without labelled anomalies and contains no such comparison [1][2].
- The cleanest positive controls come from adjacent tasks: Presto's random-init same-architecture comparison (RMSE 24.72/0.887 pretrained vs 26.75/1.063 random) [3], and S2-Agri100 (OA 70.15 vs 62.40; macro-F1 44.14 vs 40.21) [4]. Gains concentrate in the few-label regime — 67% vs 44% at 5 labels/class [5].
- The closest sugarcane work (STS-NET, 97.98% water-stress precision, 62 fields) has **no random-init control**, uses a pixel-level 80:20 split with spatial-leakage risk, and loses 7.66–17.74 points reducing 32 timesteps to 8 — a confound larger than many reported SSL gains [6][7].
- Input design can dominate pretraining: a six-band model trained from scratch beat a pretrained three-band model by 4.29 OA points [8]; spatial-context models gained +2.0 OA/+5.5 MA on PASTIS, but ranking reversed under 2018→2019 domain shift [9].
- For ~25 ha fields at 30 m, the literature supports small patches (5×5 is a published standard) **and** field aggregation to suppress boundary/mixed-pixel noise; 5×5 beat 7×7 and 64×64 in one LULC study (abstract-level) [10][11][12][13].
- A defensible comparison needs: random-init same backbone, logged collapse diagnostics, equal compute budgets, hand-crafted/rule baselines, permutation nulls at the unit of independence, and per-seed variance with significance tests [14][15][16][17][18][19]. No preregistered SSL-vs-random-init EO study was found [20].

## Background & scope
Question, boundaries and assumptions are fixed in `brief.md`. The decision at stake is a CPU-only architecture × spatial-unit sweep on 1–2 sugarcane fields from a 3-field Landsat archive, where a prior probe found retrieval above chance but inside a permutation null and AUPRC overlapping a random-init control. This report supplies the evidential standard; it does not assume a positive outcome.

## Is there a defensible SSL advantage to copy?
**Not for this task.** The review of self-supervised time-series anomaly detection explicitly excludes methods using labelled anomalies and its method tables contain no random-init-versus-pretrained comparison [1]. A peer-reviewed 2026 satellite anomaly-detection paper reports only 2–10% F1 gains over other self-supervised methods, with no random-init or supervised-from-scratch baseline in its abstract [2]. By contrast, remote-sensing classification benchmarks do report clean gains when the control is present: Presto's randomly initialized same architecture loses 9.96% mean accuracy relative to linear regression while the pretrained feature extractor gains 4.00% and fine-tuning gains 7.05% [3]; Saget et al. report 70.15 vs 62.40 OA on S2-Agri100 with the same architecture [4].

**The gain is a label-scarcity effect.** On FranceCrops, frozen SSL features with 5 labels/class reach 67% versus 44% on raw features (10 labels: 72% vs 52%) [5], and the SSL4EO review finds linear probing beats supervised learning only at 1% labels [14]. Target-domain pretraining matters more than classifier complexity: moving to target-domain unlabeled data raised S2-Agri100 OA from 70.15 to 76.84, while a linear probe already reached 74.30 [4]. This matters for the local sweep: with no meaningful label budget, the pre-registered "few-label" advantage has no analogue here.

**The sugarcane analogue is not a clean control.** STS-NET reports 97.98% water-stress precision on 62 controlled fields using a 19.2K-parameter 3D-CAE + Random Forest [6], but its 80:20 split is over pixel-series samples drawn from the same fields [7], and the temporal-depth ablation (32→8 timesteps: 97.98→90.32 water, 85.08→70.97 nitrogen, 83.47→65.73 combined) exceeds most reported SSL gains [6]. It also uses single-pixel sequences rather than spatial patches [11].

## Spatial unit: pixel, patch, or field?
Parcel-level aggregation is justified by mixed pixels and boundary effects. Gou et al. cite parcel-based classification improving OA from 0.648 to 0.721 and note that fields with fewer than ~10 10-m pixels, plus diagonal-boundary "stair-step aliasing", make pixel-wise labels unreliable; they average all in-parcel pixels [12]. A smallholder study (2,600 fields, mean 1.2 acres) explicitly criticises patch-based approaches for crossing field boundaries into neighbouring crops and instead averages per-pixel values into a per-field sequence [13]. At 30 m with ~25 ha fields, both arguments favour field aggregation as a reference arm.

Spatial context can still help. UTAE beat pixel-sequence methods on PASTIS by +2.0 OA/+5.5 MA with folds separated by ≥1 km to prevent leakage [9]. But the same benchmark's DENETHOR 2018→2019 shift reversed the ordering: simple nearest-centroid methods beat complex models [9]. Presto's S2-Agri100 parcels use a 5×5 patch and report ~50% relative accuracy gain over random init [10]; a 2026 LULC study reports 5×5 beating 7×7 and 64×64 (97.6% accuracy, kappa 0.95) [11] [single source, abstract only]. **Conclusion for the sweep:** patch size is a confound to be screened, not a hypothesis to be presumed; include 1, 3, 5, 7, 9, 11 and field-aggregate arms with equal compute, and read any within-year winner only on a cross-year split.

## What makes a positive number defensible
1. **Same-architecture random init is mandatory.** SSL4EO benchmarks explicitly include randomly initialized encoders of the same ResNet-18 backbone alongside fully supervised learning [14]. Presto's task-dependence is itself a warning: on TreeSatAI S1, random init slightly beat pretraining [10].
2. **Equal compute, or the claim is about convergence.** ImageNet pretraining speeds early convergence but does not necessarily improve final accuracy or regularisation [15]; a win at unequal epochs is not a representation-quality result.
3. **Collapse must be logged.** SSL4EO had to stop SimSiam at 10 epochs because its loss became unstable [14]; VICReg provides per-dimension variance plus covariance decorrelation as an inspectable diagnostic [16], and Barlow Twins turns collapse avoidance into an identity cross-correlation target [17].
4. **Backbone and inputs are confounds.** Architecture choices can dominate pretext-task choices [18], and six-band scratch beats three-band pretrained by 4.29 points [8] — so architecture, channel set and patch size must be held fixed when attributing a difference to SSL.
5. **Small-n needs hand-crafted baselines and group-level nulls.** Under leakage-safe Leave-One-Region-Out on 14 regions, hand-crafted LBP+SVM (balanced accuracy 0.79) beat every deep and SSL embedding; a group-level permutation null (500 of 3,003 assignments) gave p=0.008 [19]. The EO anomaly-detection precedent pairs classical baselines, an independent pseudo-ground-truth modality, and a permutation test (77.0% ROC-AUC vs ~50% on 300 series, three AOIs) [20].
6. **Report per-seed variance.** A Mamba-vs-Transformer benchmark with a shared pipeline reported a 1.3-point gap but explicitly declined a definitive ranking for lack of per-seed variance, significance testing, and capacity/compute controls [21]. No preregistered SSL-vs-random-init EO comparison was found [20].

## Implication for the local sweep
The pre-registered gates should judge a model on: equal-compute random-init control, rule/feature baseline, collapse diagnostics, a farm×year permutation null, and ≥3 seeds with CIs — and must allow a null result. The literature supplies no precedent for asserting SSL advantage a priori on this task, especially while the field-ID provenance is unresolved.

## Open questions
- Does the +1 field-ID alias hold across all 490 farms, or only the sampled three? (Local evidence covers three; no canonical re-index was performed.)
- Can an independent agronomic modality be bound for L4 agreement? The EO precedent relies on one never used by the model [20].
- Which patch size survives a cross-year split rather than only within the 2013–2026 archive?

## Sources
[1] Survey of self-supervised time-series anomaly detection (2025) — https://arxiv.org/html/2501.15196v1 (accessed 2026-11-25)
[2] Satellite-image anomaly detection, Neural Computing & Applications 38 (2026) — https://link.springer.com/article/10.1007/s00521-025-11746-w (accessed 2026-11-25)
[3] Presto v2 (2023) — https://arxiv.org/html/2304.14065v2 (accessed 2026-11-25)
[4] Saget et al., S2-Agri100 / Terrabytes 2025 — https://arxiv.org/html/2506.18587v1 (accessed 2026-11-25)
[5] FranceCrops few-label results — https://arxiv.org/html/2506.18587v1 (accessed 2026-11-25)
[6] STS-NET sugarcane study (2026) — https://arxiv.org/html/2607.18791v1 (accessed 2026-11-25)
[7] STS-NET split protocol — https://arxiv.org/html/2607.18791v1 (accessed 2026-11-25)
[8] SAR-TS-BERT (2026) — https://arxiv.org/html/2610.41728v1 (accessed 2026-11-25)
[9] Vincent et al., UTAE / PASTIS-DENETHOR (2024) — https://arxiv.org/html/2303.12533v2 (accessed 2026-11-25)
[10] Presto v4 (2024) — https://arxiv.org/html/2304.14065v4 (accessed 2026-11-25)
[11] Sentinel-2 patch-size study (2026, abstract) — https://link.springer.com/chapter/10.1007/978-3-032-21013-5_16 (accessed 2026-11-25)
[12] Gou et al., Scientific Reports (2026) — https://www.nature.com/articles/s41598-026-68078-5 (accessed 2026-11-25)
[13] Khan et al., Sensors (2023) — https://pmc.ncbi.nlm.nih.gov/articles/PMC9967001/ (accessed 2026-11-25)
[14] SSL4EO review/benchmark (2022) — https://ar5iv.labs.arxiv.org/html/2206.13188 (accessed 2026-11-25)
[15] He et al., pretraining vs convergence (2018) — https://arxiv.org/abs/1811.08883 (accessed 2026-11-25)
[16] VICReg (ICLR 2022) — https://arxiv.org/abs/2105.04906 (accessed 2026-11-25)
[17] Barlow Twins (ICML 2021) — https://arxiv.org/abs/2103.03230 (accessed 2026-11-25)
[18] SSL backbone design (2019) — https://arxiv.org/abs/1901.09005 (accessed 2026-11-25)
[19] Small-n negative result with group permutation (2026) — https://arxiv.org/abs/2609.28567 (accessed 2026-11-25)
[20] T-SAR-JEPA, IGARSS 2026 / DFC 2026 — https://arxiv.org/abs/2606.05700 (accessed 2026-11-25)
[21] Mamba-vs-Transformer benchmark (2025/2026) — https://arxiv.org/abs/2509.00141 (accessed 2026-11-25)
