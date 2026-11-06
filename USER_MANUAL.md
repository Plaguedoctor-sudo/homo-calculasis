# Whole-Body Clinical Workstation: Comprehensive User Manual
**Computational Psychiatry & Metabolic Neuroscience Decision-Support System**

*Authored for Practicing Physicians, Clinical Psychiatrists, Endocrinologists, and Translational Neuroscientists*  
*Standardized for Indian NABL Diagnostic Reference Standards & South Asian Metabolic Phenotypes*

---

## Table of Contents
1. [Clinical Rationale & Paradigm Shift](#1-clinical-rationale--paradigm-shift)
2. [Whole-Body Physiological Architecture](#2-whole-body-physiological-architecture)
3. [Indian Clinical Reference Standards & Lab Units](#3-indian-clinical-reference-standards--lab-units)
4. [Using the Physician Workstation (http://localhost:8050)](#4-using-the-physician-workstation)
   - [4.1 Patient Simulation & Live Labs](#41-patient-simulation--live-labs)
   - [4.2 The 12-Parameter Live Downstream Cascade](#42-the-12-parameter-live-downstream-cascade)
   - [4.3 Creating & Adding Custom Parameters / Clamps](#43-creating--adding-custom-parameters--clamps)
   - [4.4 Symptom Severity Triage & Treatment Forecaster](#44-symptom-severity-triage--treatment-forecaster)
   - [4.5 Formal Doctor's Consultation & Prescription Slip](#45-formal-doctors-consultation--prescription-slip)
   - [4.6 Organ Body Map & Causal Graph](#46-organ-body-map--causal-graph)
   - [4.7 Bayesian Inversion (NPE) & Optimal Trial Design (BOED)](#47-bayesian-inversion-npe--optimal-trial-design-boed)
5. [Clinical Case Studies & Presets](#5-clinical-case-studies--presets)
6. [CLI & Server Operations](#6-cli--server-operations)
7. [API Reference](#7-api-reference)

---

## 1. Clinical Rationale & Paradigm Shift

Major Depressive Disorder (MDD), chronic fatigue syndromes, and treatment-resistant mood disorders are heterogeneous whole-body multi-system disorders. Traditional DSM-5 diagnostic approaches rely entirely on subjective symptom clusters, frequently failing to identify the somatic and biochemical root causes of treatment failure.

In clinical practice—especially within South Asian and Indian populations—antidepressant non-response (up to 50–60% with first-line SSRI monotherapy) is frequently driven by unrecognized peripheral metabolic and nutritional bottlenecks:
1. **Severe Iron & B12 Depletion:** Prevalent in predominantly vegetarian diets; starves the rate-limiting enzymes Tryptophan Hydroxylase (TPH2) and Tyrosine Hydroxylase (TH).
2. **South Asian Metabolic Phenotype:** High visceral adiposity and insulin resistance at normal BMI (HOMA-IR > 2.5), causing astrocytic bioenergetic failure and impaired glutamate clearance (EAAT2 downregulation).
3. **Immune-Kynurenine Diversion:** Low-grade systemic endotoxemia (elevated LPS / leaky gut) and subclinical inflammation (hs-CRP > 3.0 mg/L) activate IDO1, diverting tryptophan away from serotonin towards neurotoxic Quinolinic Acid (QA).
4. **HPA Axis Glucocorticoid Resistance:** Melancholic phenotypes characterized by overnight 1mg Dexamethasone non-suppression (post-DEX cortisol ≥ 1.8 µg/dL).

The **Whole-Body Clinical Workstation** bridges this gap by mechanistically coupling routine peripheral blood tests to a continuous dynamical model of central neurotransmission, mitochondrial bioenergetics, glial glutamate homeostasis, and attractor mood dynamics.

---

## 2. Whole-Body Physiological Architecture

The workstation models human physiology across interconnected multi-scale biological compartments governed by coupled Ordinary Differential Equations (ODEs):

```
+----------------------------------------------------------------------------+
|                         EXPOSOME & NUTRITION AXIS                          |
|  - Dietary Fe2+, B12, Zn, Se, Mg, Folate   - Particulate Air PM2.5        |
|  - Fasting Glycemia & Serum Insulin         - Circadian Lux & Retinal Melan |
+----------------------------------------------------------------------------+
                                     |
                                     v
+------------------------+  +------------------------+  +--------------------+
|  INTESTINAL BARRIER    |  |  IMMUNE / KYNURENINE   |  | METABOLIC / MITO   |
| - Tight Junction Leak  |->| - IDO1 Activation      |  | - Insulin Resist.  |
| - LPS Endotoxin Influx |  | - QA/KA Neurotoxicity  |  | - Astrocytic EAAT2 |
| - Vagal Afferent Tone  |  | - Cytokines (IL-6/TNF) |  | - ATP Bioenergetics|
+------------------------+  +------------------------+  +--------------------+
                                     |
                                     v
+----------------------------------------------------------------------------+
|                 CENTRAL TRIPARTITE SYNAPSE & NEUROENDOCRINE                |
|  - TPH2 5-HT Synthesis (Fe/BH4)        - Astrocytic EAAT2 Glutamate Uptake|
|  - TH Dopamine Synthesis (Fe/BH4)      - HPA Stress & Cortisol Resistance |
|  - Brain Free T3 Delivery              - SWS Glymphatic Clearance (AQP4)  |
+----------------------------------------------------------------------------+
                                     |
                                     v
+----------------------------------------------------------------------------+
|                       NON-LINEAR ATTRACTOR DYNAMICS                        |
|   dU/dm: Bistable Double-Well Potential [Euthymic (+1) vs Sinking (-1)]    |
|   Allostatic Load Strain (L) + Descending Pain Gating (DNIC Threshold)     |
+----------------------------------------------------------------------------+
```

### Mathematical Attractor Formulation
The patient's clinical mood trajectory $m(t)$ is modeled as gradient descent on a double-well potential modulated by trophic drives and allostatic stress:

$$\frac{dm}{dt} = -\frac{\partial U(m)}{\partial m} = -(m^3 - m - \text{NetDrive})$$

Where:
$$\text{NetDrive} = \text{AllostaticStress} - \text{TrophicSupport} + 0.20$$
$$\text{TrophicSupport} = 0.35 \cdot [5\text{-HT}] + 0.35 \cdot [\text{DA}] + 0.25 \cdot [\text{ATP}] + 0.15 \cdot [T_3^{\text{brain}}]$$
$$\text{AllostaticStress} = 0.30 \cdot L + 0.25 \cdot [\text{IL-6}] + 0.15 \cdot \left(\frac{\text{QA}}{\text{KA}}\right) + 0.20 \cdot \text{Tax}_{\text{psy}}$$

* When $\text{NetDrive} > 0$, the euthymic well ($+1.0$) disappears via a saddle-node bifurcation, causing the patient to **sink catastrophically into the depressive minimum ($-1.0$)**.

---

## 3. Indian Clinical Reference Standards & Lab Units

All lab parameters are calibrated to **Indian National Accreditation Board for Testing and Calibration Laboratories (NABL)** standards (Dr Lal PathLabs, SRL Diagnostics, AIIMS New Delhi):

| Parameter | Indian Unit | Normal Reference Range | Deficient / Risk Level | Clinical Mechanism |
|---|---|---|---|---|
| **Serum Ferritin** | `ng/mL` | 30 – 300 | **< 30** (Critical < 15) | Rate-limiting cofactor for Tyrosine & Tryptophan Hydroxylases (TH/TPH2). Low levels stall monoamine synthesis. |
| **Vitamin B12** | `pg/mL` | 211 – 911 | **< 211** (Severe < 150) | Impairs 1-carbon SAMe methylation cycle; causes BH4 depletion and homocysteine neurotoxicity. |
| **25-OH Vitamin D3** | `ng/mL` | 30 – 100 | **< 20** (Insufficient < 30) | Downregulates BDNF transcription; exacerbates chronic neuroinflammation. |
| **Serum Zinc** | `µg/dL` | 70 – 120 | **< 70** | Essential for Kynurenine Aminotransferase (KAT); low zinc shifts pathway to neurotoxic Quinolinic Acid. |
| **Serum Selenium** | `µg/L` | 70 – 150 | **< 70** | Glutathione Peroxidase (GPx) antioxidant cofactor; low selenium elevates mitochondrial ROS. |
| **Serum Magnesium** | `mg/dL` | 1.8 – 2.4 | **< 1.8** | Voltage-dependent NMDA channel blocker; deficiency causes intracellular calcium excitotoxicity. |
| **Fasting Blood Sugar (FBS)** | `mg/dL` | 70 – 99 | **≥ 100** (Prediabetes), **≥ 126** (Diabetes) | Drives systemic advanced glycation end-products (AGEs) and neurovascular inflammation. |
| **Fasting Serum Insulin** | `µIU/mL` | 2.6 – 24.9 | **> 15.0** (High fasting insulin) | Hyperinsulinemia promotes peripheral insulin resistance and astrocytic bioenergetic compromise. |
| **Calculated HOMA-IR** | Ratio | < 1.5 (Normal) | **> 2.5** (Insulin Resistant) | Calculated as `(FBS × Insulin) / 405`. Impairs astrocytic EAAT2 glutamate clearance. |
| **TSH** | `µIU/mL` | 0.4 – 4.5 | **> 4.5** (Hypothyroid) | Diminishes brain free T3 delivery, slowing neuronal firing and mitochondrial ATP generation. |
| **Overnight 1mg DEX Cortisol** | `µg/dL` | < 1.8 (Suppressed) | **≥ 1.8** (Non-Suppressor) | Confirms central glucocorticoid receptor down-regulation and melancholic HPA axis hyperdrive. |
| **hs-CRP** | `mg/L` | < 1.0 (Low risk) | **> 3.0** (High cardiovascular & psychiatric risk) | Activates hepatic and endothelial IDO1, provoking monoamine depletion. |
| **Interleukin-6 (IL-6)** | `pg/mL` | < 5.0 | **≥ 5.0** (Elevated) | Crosses blood-brain barrier to trigger microglial activation and anhedonia. |
| **HRV RMSSD** | `ms` | 35 – 70 | **< 35** (Autonomic depletion) | Index of cardiovagal parasympathetic buffering; low RMSSD opens the spinal DNIC pain gate. |
| **Endotoxin (LPS)** | `EU/mL` | < 0.20 | **≥ 0.20** (Leaky gut) | Translocates across damaged intestinal tight junctions into portal circulation. |

---

## 4. Using the Physician Workstation

Open your web browser and navigate to:
```
http://localhost:8050
```

### 4.1 Patient Simulation & Live Labs
1. Switch to the **🏥 Patient & Indian Labs** tab.
2. Select a profile from the **Clinical Biotype Presets** dropdown (e.g. *Indian Vegetarian Anhedonia*).
3. **Drag any lab slider:** Notice how the value and the NABL reference badge (`CRITICAL DEFICIENT`, `LOW RISK`, `INSULIN RESISTANT`, etc.) update immediately.
4. **Instant Calculation:** Every slider movement recalculates the whole-body simulation within **60 milliseconds**.

### 4.2 The 12-Parameter Live Downstream Cascade
Positioned in the right-hand panel, this grid displays 12 interconnected variables that change dynamically whenever you move any slider or add custom parameters:
* 🧬 **Serotonin Flux (5-HT)** (Normal $\ge 0.800$, Depleted $< 0.500$)
* 🎯 **Dopaminergic Tone** (Normal $\ge 0.350$, Depleted $< 0.220$)
* ⚡ **ATP Bioenergetics** (Normal $\ge 0.850$, Deficient $< 0.650$)
* ☣️ **QA/KA Excitotoxicity Ratio** (Safe $\le 2.20$, High Risk $> 3.00$)
* 📉 **Allostatic Load $L$** (Compensated $\le 0.400$, Severe $> 0.750$)
* 🧭 **Mood Well State $m$** (Euthymic $> 0$, Sinking $\le 0$)
* 🩸 **HOMA-IR Index** (Sensitive $\le 1.50$, Resistant $> 2.50$)
* 🫀 **Vagal Efferent Tone** (Normal $\ge 0.800$, Exhaustion $< 0.400$)
* 🛡️ **Systemic Cytokine Drive (IL-6)** (Normal $\le 2.00$, Inflammatory $> 3.00$)
* ⚡ **DNIC Pain Gate Threshold** (Normal $\ge 0.600$, Hyperalgesic $< 0.350$)
* 🧠 **Astrocytic EAAT2 Clearance** (Intact $\ge 1.50$, Impaired $< 1.00$)
* 🌙 **Glymphatic SWS Density** (Optimal $\ge 0.65$, Disrupted $< 0.40$)

### 4.3 Creating & Adding Custom Parameters / Clamps
Physicians can introduce novel lab biomarkers, investigative assays, or experimental drug provocations:
1. In the left column under the lab sliders, find the dashed box labeled **⭐ Custom Parameters & Clinical Clamps**.
2. Click **+ Add Parameter**.
3. Choose a **Clinical Template** (e.g. *Serum Folate*, *TNF-Alpha*, *Serum Zonulin*, *Cortisol CAR*, *Exercise Volume*, *Metformin Therapy*, *Esketamine Infusion*) or type custom values:
   * **Parameter Name** & **Unit** (e.g. `Serum Folate`, `ng/mL`)
   * **Min**, **Max**, **Step**, and **Initial Value**
   * **Target Physiological Axis** (e.g. `Monoamine Serotonin Synthesis`, `Astrocytic EAAT2 Clearance`, `Mitochondrial Bioenergetics`)
   * **Coupling Direction** (`Direct (+)` or `Inverse (-)`)
4. Click **Save & Couple to Simulation**.
5. The custom parameter immediately receives an interactive slider. Dragging this slider dynamically shifts the entire downstream cascade, symptom meters, and treatment predictions.
6. Click **✕** on any custom card to remove it from the model.

### 4.4 Symptom Severity Triage & Treatment Forecaster
* **Symptom Gauges (0–10 Scale):**
  * *Anhedonia & Blunting:* Driven by dopaminergic synthesis deficit.
  * *Bioenergetic Fatigue:* Driven by mitochondrial ATP flux and brain T3 starvation.
  * *Cognitive Brain Fog:* Driven by combined fatigue, anhedonia, and QA/KA neurotoxicity.
  * *Somatic Pain Amplification:* Driven by cardiovagal DNIC gate collapse and inflammatory hyperalgesia.
  * *Sleep Disruption (SWS):* Driven by glymphatic slow-wave sleep downscaling.
  * *Sinking Risk Percentage (0–100%):* Quantifies risk of falling into the depressive attractor basin.
* **Treatment Response Forecaster (% Probability):**
  * **SSRI Monotherapy:** Penalized when Ferritin < 30 ng/mL, B12 < 200 pg/mL, or DEX non-suppression is present.
  * **Left DLPFC 10 Hz rTMS:** Upregulates astrocytic EAAT2; predicted highly effective when SSRIs fail or glutamate excitotoxicity is elevated.
  * **NMDA Receptor Antagonists (Ketamine / Esketamine):** Predicted response increases with elevated QA/KA ratio.
  * **Micronutrient Repletion:** Indicated (> 85% response) when cofactor deficiencies are identified.

### 4.5 Formal Doctor's Consultation & Prescription Slip
1. Switch to the **📋 Doctor's Consultation & Rx** tab.
2. The workstation automatically compiles:
   * Patient diagnostic summary (HAM-D estimate, HOMA-IR, DEX suppression status).
   * Key clinical contraindications and alerts.
   * Required NABL laboratory investigations checklist.
   * Evidence-based prescription table (Dosage, frequency, duration, clinical notes).
   * Non-pharmacological and interventional protocols (rTMS, Dietary, Phototherapy).
3. Click **🖨️ Print Consultation Slip** to generate a clean PDF/paper requisition, or **📋 Copy Note to Clipboard** for pasting into EHR systems.

### 4.6 Organ Body Map & Causal Graph
* Switch to the **🫀 Organ Body Map & Cascades** tab.
* Toggle between **Anatomy Layout** (spatially arranged by organ: Gut, Liver, Thyroid, Heart, Adrenals, Hippocampus, Prefrontal Cortex) and **Topology Layout** (hierarchical causal graph).
* Select a **Clinical Cascade** from the dropdown (*Nutritional-Monoamine Collapse*, *Metabolic Glial Failure*, *Gut-Brain Inflammatory Diversion*, *Melancholic HPA Glucocorticoid Resistance*) to highlight active pathways.
* Click any node to open the **Node Inspector**, which displays biochemical reactions, rate constants, and clinical interpretations.

### 4.7 Bayesian Inversion (NPE) & Optimal Trial Design (BOED)
* **Amortized NPE Latent Inversion:** Click **Run Inversion** on the patient tab to run neural posterior estimation, inferring all 16 micro-kinetic parameters from the patient's observable lab panel with posterior contraction ($\eta$).
* **Dynamic Provocation Design (Tab 4):** Click **Compute Optimal Provocation** to identify which biological clamp (e.g. Oral Tryptophan Challenge, Acute Light Pulse, Dexamethasone Suppression) delivers the highest Expected Information Gain (EIG in nats) for resolving diagnostic ambiguity.
* **Clinical Trial Validation (Tab 5):** Evaluates parameter recovery against synthetic trials (EMBARC benchmarks), verifying $R^2 > 0.75$, calibration slope, and drug-placebo divergence.

---

## 5. Clinical Case Studies & Presets

### Case 1: Indian Vegetarian Anhedonia & Pseudo-Resistance
* **Labs:** Ferritin 14 ng/mL, B12 140 pg/mL, Vit D 16 ng/mL, Zn 52 µg/dL, DEX 3.2 µg/dL.
* **Pathology:** Severe monoamine synthesis stall. SSRI monotherapy has a predicted response rate of only **6.0%**.
* **Workstation Protocol:**
  1. Co-prescribe Ferrous Ascorbate (100mg elemental Fe) + Vitamin C OD.
  2. Methylcobalamin 1500 mcg + L-Methylfolate 7.5 mg OD.
  3. Cholecalciferol 60,000 IU weekly for 8 weeks.
  4. Defer or co-administer SSRI only after 4 weeks of cofactor repletion.

### Case 2: South Asian Metabolic-Glial Depressive Phenotype
* **Labs:** FBS 132 mg/dL, Insulin 22 µIU/mL (HOMA-IR 7.17), Ferritin 60 ng/mL, hs-CRP 4.8 mg/L.
* **Pathology:** Astrocytic insulin resistance impairs EAAT2 glutamate transporters, causing excitotoxicity, severe brain fog, and SSRI resistance.
* **Workstation Protocol:**
  1. Low-glycemic Mediterranean/South Asian dietary modification.
  2. Metformin SR 500–1000 mg OD or Myo-Inositol 2000 mg BD.
  3. Left DLPFC 10 Hz rTMS to directly upregulate EAAT2 expression.

### Case 3: Inflammatory Kynurenine Excitotoxic Syndrome
* **Labs:** hs-CRP 5.6 mg/L, IL-6 8.5 pg/mL, LPS 1.20 EU/mL, QA/KA 3.8.
* **Pathology:** Leaky gut endotoxemia drives hepatic/endothelial IDO1. Tryptophan is shunted away from serotonin into neurotoxic Quinolinic Acid.
* **Workstation Protocol:**
  1. Intestinal barrier support: Zinc Carnosine + Glutamine.
  2. NMDA receptor antagonism: Ketamine / Esketamine protocol.
  3. Anti-inflammatory lifestyle: Omega-3 fatty acids (EPA > 1000 mg/d).

---

## 6. CLI & Server Operations

### Starting the Workstation Server
Run the automated startup script from the repository root:
```bash
./run_server.sh
```
This launches the server on port `8050` with `allow_reuse_address` enabled.

### Specifying Custom Host or Port
```bash
.venv/bin/python3 ui/server.py 8080 127.0.0.1
```

### Running the Test Suite
The full whole-body modeling suite includes 68 unit, mathematical invariant, and clinical validation tests:
```bash
.venv/bin/pytest
```
*Expected output: `68 passed in ~10s`.*

---

## 7. API Reference

All endpoints accept and return UTF-8 JSON with CORS enabled:

### `GET /api/health`
Health check and endpoint inventory.
```json
{ "status": "active", "simulator": "WholeBodyPatient", "dim": 16 }
```

### `POST /api/simulate`
Simulates patient trajectory over time and computes all clinical readouts.
* **Payload:**
  ```json
  {
    "ferritin": 18.0,
    "b12": 150.0,
    "vit_d": 16.0,
    "zinc": 55.0,
    "selenium": 65.0,
    "magnesium": 1.7,
    "fbs": 108.0,
    "fasting_insulin": 18.5,
    "tsh": 3.8,
    "dex_suppression": 3.2,
    "hs_crp": 3.4,
    "il6": 4.5,
    "hrv_rmssd": 22.0,
    "lps": 0.60,
    "intervention": "repletion",
    "custom_parameters": [
      {
        "name": "Serum Folate",
        "value": 16.0,
        "baseline": 6.0,
        "min": 1.0,
        "max": 25.0,
        "target_axis": "serotonin_monoamine",
        "direction": "positive"
      }
    ],
    "steps": 1440
  }
  ```
* **Returns:** `final_m`, `final_L`, `final_qa_ka`, `basin_status`, `downstream_cascade` (12 variables), `symptoms`, `treatment_predictions`, `clinical_alerts`, `doctor_rx`, and `trajectory` time-series.

### `POST /api/patient/infer`
Inverts observable labs into posterior estimates of 16 latent micro-kinetic parameters.
* **Payload:** `{"profile": "indian_veg_anhedonia"}`
* **Returns:** `inferred`, `true`, and `contraction` per parameter.

### `POST /api/design`
Bayesian Optimal Experimental Design (BOED) for selecting optimal diagnostic provocation clamps.

### `GET /api/biotypes`
Extracts empirical Bayes patient clusters and effective rank of psychiatric variance.

### `GET /api/validation`
Evaluates prospective synthetic clinical trial benchmarks (T1–T4 recovery).

### `GET /api/export`
Downloads the last simulation's multi-channel trajectory as a CSV file (`patient_trajectory.csv`).

---

*Whole-Body Computational Psychiatry Workstation © 2026 Dr. Nitin Sonare. Distributed under the MIT License.*
