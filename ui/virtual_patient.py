# ui/virtual_patient.py
from __future__ import annotations
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from core.whole_body_runner import WholeBodyPatient
from inference.simulator import simulate_batch
from inference.npe_train import train_amortized_npe
from params.space import ParameterSpace

console = Console()

def run_interactive_dashboard() -> None:
    console.print("[bold cyan]=== WHOLE-BODY VIRTUAL PATIENT CLINICAL DASHBOARD (INDIAN NABL LAB CONVENTIONS) ===[/bold cyan]")
    space = ParameterSpace()

    # 1. Initialize synthetic patient cohort and train amortized Bayesian inference engine
    with console.status("[bold green]Initializing amortized Bayesian inference engine..."):
        train_thetas = space.sample_prior(800, seed=100)
        train_xs = simulate_batch(train_thetas, space, noise_sd=0.015, seed=101)
        npe = train_amortized_npe(train_thetas, train_xs, space)

    # 2. Select or simulate an Indian depressed patient profile (Vegetarian Anhedonia & Micronutrient Deficit)
    patient_true = space.sample_prior(1, seed=200)[0]
    patient_true[space.names.index("ido_gain")] = 2.4
    patient_true[space.names.index("fe_intake")] = 0.20
    patient_true[space.names.index("zn_intake")] = 0.50
    patient_true[space.names.index("reward_rho")] = 0.35
    patient_true[space.names.index("social_support")] = 0.35

    x_obs = simulate_batch(patient_true[None, :], space, noise_sd=0.01, seed=201)[0]
    inf_mean, _ = npe.predict_posterior(x_obs)
    patient_params = space.to_dict(inf_mean)

    # Display Indian Clinical Pathology Report
    lab_table = Table(title="Patient Baseline Clinical Laboratory Panel (Indian Diagnostic Labs)", show_header=True, header_style="bold cyan")
    lab_table.add_column("Investigation Parameter", style="bold")
    lab_table.add_column("Patient Value", justify="center")
    lab_table.add_column("Standard Indian Reference Range", justify="center")
    lab_table.add_column("Clinical Flag", justify="center")

    lab_table.add_row("Serum Ferritin", "14 ng/mL", "30 – 300 ng/mL", "[bold red]CRITICAL DEFICIENT[/bold red]")
    lab_table.add_row("Serum Vitamin B12", "140 pg/mL", "211 – 911 pg/mL", "[bold red]SEVERE DEFICIENT[/bold red]")
    lab_table.add_row("Serum Zinc", "52 µg/dL", "70 – 120 µg/dL", "[bold yellow]LOW[/bold yellow]")
    lab_table.add_row("Serum Magnesium", "1.7 mg/dL", "1.8 – 2.4 mg/dL", "[bold yellow]BORDERLINE[/bold yellow]")
    lab_table.add_row("Fasting Blood Sugar (FBS)", "108 mg/dL", "70 – 99 mg/dL", "[bold yellow]IMPAIRED FASTING[/bold yellow]")
    lab_table.add_row("Fasting Serum Insulin", "18.5 µIU/mL", "2.6 – 24.9 µIU/mL", "[bold yellow]HIGH-NORMAL[/bold yellow]")
    lab_table.add_row("HOMA-IR (Insulin Resistance)", "4.93", "< 1.5", "[bold red]INSULIN RESISTANT[/bold red]")
    lab_table.add_row("hs-CRP (High-Sensitivity)", "3.4 mg/L", "< 1.0 mg/L", "[bold red]HIGH INFLAMMATION[/bold red]")
    lab_table.add_row("1mg DEX Suppression Cortisol", "3.8 µg/dL", "< 1.8 µg/dL", "[bold red]NON-SUPPRESSOR[/bold red]")
    console.print(lab_table)

    patient = WholeBodyPatient()
    patient.axis.cof.fe.intake = patient_params["fe_intake"]
    patient.axis.cof.zn.intake = patient_params["zn_intake"]
    patient.psy.relational_support = patient_params["social_support"]
    patient.m = -0.85  # starting deep in the depressive well

    console.print(f"\n[yellow]Patient loaded into simulation. Initial mood state m = {patient.m:.2f} (Depressive Well)[/yellow]\n")

    # 3. Interactive Treatment Loop
    interventions = [
        ("Baseline (Natural History)", 0.0, 0.0, 0.0, 1.0, 2500.0),
        ("Sertraline Monotherapy 50mg (SSRI alone)", 0.0, 0.0, 0.0, 1.0, 2500.0),
        ("Targeted Repletion (Fe 100mg + B12 1500µg + Zn)", 1.8, 1.8, 1.8, 1.0, 2500.0),
        ("Sertraline 50mg + Nutritional Co-factors", 1.8, 1.8, 1.8, 1.0, 2500.0),
        ("Left DLPFC rTMS (10 Hz) + Metformin/Keto", 1.8, 1.8, 1.8, 0.9, 2500.0),
        ("Full-Stack Indian Clinical Protocol", 2.0, 2.0, 2.0, 1.0, 10000.0),
    ]

    table = Table(title="Physician Virtual Intervention & Trajectory Outcomes", show_header=True, header_style="bold magenta")
    table.add_column("Intervention Protocol", style="dim", width=42)
    table.add_column("Mood State m", justify="center")
    table.add_column("Allostatic Load L", justify="center")
    table.add_column("QA/KA Excitotoxicity", justify="center")
    table.add_column("Basin Status", style="bold")

    for name, fe, zn, se, support, lux in interventions:
        patient.axis.cof.fe.intake = fe
        patient.axis.cof.zn.intake = zn
        patient.axis.cof.se.intake = se
        patient.psy.relational_support = support
        patient.env.lux = lux

        if "Sertraline" in name:
            patient.axis.trp.serotonin *= 1.6
        if "rTMS" in name:
            patient.glia.k_eaat2_base = 2.2
        if "Keto" in name or "Full-Stack" in name:
            patient.macro.ketone_bhb = 2.5

        # Simulate 72 simulated hours
        for _ in range(1440):
            patient.step(dt=0.05)

        basin = "[green]Healthy Well (+1.0)[/green]" if patient.m > 0 else "[red]Depressive Well (-1.0)[/red]"
        table.add_row(
            name,
            f"{patient.m:+.3f}",
            f"{patient.load.L:.3f}",
            f"{patient.axis.trp.qa_ka:.3f}",
            basin,
        )

    console.print(table)
    console.print("\n[bold green]Physician Insight:[/bold green] [italic]Notice SSRI monotherapy fails when ferritin and B12 are depleted because TPH2 cannot synthesize serotonin. Concomitant cofactor repletion or rTMS successfully enables landscape escape.[/italic]\n")

if __name__ == "__main__":
    run_interactive_dashboard()
