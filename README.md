# Homo-Calculasis - Whole-Body Computational Psychiatry and Metabolic Workstation

> [!IMPORTANT]
> Development status: Under active development.

Homo-Calculasis is a whole-body clinical decision-support and computational neuroscience modeling platform. It mechanistically couples routine peripheral blood biomarkers to continuous dynamical models of central neurotransmission, glial glutamate homeostasis, mitochondrial bioenergetics, and mood attractor dynamics.

---

## What it can do

- **Multi-scale ordinary differential equations (ODE) modeling**:
  - **Exposome and nutritional axis**: Simulates dietary micronutrient levels (serum ferritin, elemental iron, active B12, folate, zinc, selenium, magnesium), particulate matter air exposure (PM2.5), and circadian lux exposure.
  - **Intestinal barrier dynamics**: Models gut tight-junction permeability, zonulin upregulation, lipopolysaccharide (LPS) endotoxemia, and vagal afferent signaling.
  - **Immune and kynurenine pathway**: Models indolamine-2,3-dioxygenase (IDO1) enzyme induction by pro-inflammatory cytokines (IL-6, TNF-alpha, hs-CRP), tryptophan diversion away from serotonin, and the balance between neurotoxic quinolinic acid (QA) and neuroprotective kynurenic acid (KA).
  - **Metabolic and mitochondrial bioenergetics**: Simulates insulin resistance (HOMA-IR), astrocytic bioenergetic failure, EAAT2 glial glutamate transporter downregulation, and neuronal ATP generation.
  - **Central neurotransmitter and endocrine dynamics**: Tracks central serotonin (5-HT), dopamine (DA), norepinephrine (NE), brain-derived neurotrophic factor (BDNF), and HPA-axis cortisol feedback including 1mg dexamethasone suppression responses.

- **Interactive physician workstation (`http://localhost:8050`)**:
  - Live 12-parameter downstream cascade computing dynamic physiological trajectories.
  - Real-time parameter clamping for simulated clinical interventions (e.g., intravenous iron sucrose repletion, GLP-1 / insulin sensitization, anti-inflammatory therapies).
  - Symptom severity triage and treatment response forecasting across melancholic, atypical, and inflammatory depressive phenotypes.
  - Interactive anatomical organ body map and directed causal network graph.
  - Formal clinical consultation and prescription slip generator with drug-biomarker and drug-nutrient interaction warnings.

- **Statistical inference and experimental design**:
  - Bayesian Neural Posterior Estimation (NPE) for estimating latent biological parameters from routine clinical blood panels.
  - Bayesian Optimal Experimental Design (BOED) prioritizing the most informative next laboratory test to minimize diagnostic uncertainty.

- **South Asian and Indian NABL clinical calibration**:
  - Standardized against Indian NABL laboratory reference intervals and clinical cutoff standards.
  - Accounts for South Asian metabolic phenotypes (insulin resistance at normal BMI, high vegetarian iron and cobalamin deficiency rates).

---

## What it cannot do

- **Is not a certified medical device**: Homo-Calculasis is a computational research and clinical decision-support framework; it is not certified as a medical device by the FDA, CDSCO, or CE. It does not provide autonomous clinical diagnoses or legally binding prescriptions.
- **Does not replace licensed medical practitioners**: All simulation results, risk stratifications, and suggested therapeutic candidates require independent clinical evaluation and verification by a licensed physician.
- **Does not process raw genomic sequencing files**: The system models physiological pathways from biochemical and laboratory values; it does not parse raw next-generation sequencing data (e.g., FASTQ, BAM, or VCF files).
- **Does not ingest direct real-time continuous sensor streams**: Values are input via laboratory test panels or batch profiles rather than direct real-time telemetry from continuous glucose monitors (CGM) or wearable EEG headsets.
- **Simulations are mathematical models**: Biological ODE dynamics represent simplified literature-derived approximations; they cannot account for rare unmodeled genetic variants, undocumented drug interactions, or idiosyncratic individual immune reactions.

---

## Operations and launch

### Prerequisites
- Python 3.10+
- Virtual environment with dependencies installed

### Running the workstation
```bash
# Clone the repository
git clone https://github.com/Plaguedoctor-sudo/homo-calculasis.git
cd homo-calculasis

# Install dependencies and start server
./run_server.sh
```

Access the physician workstation at: http://localhost:8050

For in-depth clinical parameters, ODE mathematical derivations, and case studies, refer to [USER_MANUAL.md](USER_MANUAL.md).

---

## License

Refer to repository license file for terms and conditions.
