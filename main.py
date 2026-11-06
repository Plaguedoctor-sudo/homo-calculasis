# main.py
from __future__ import annotations
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest
from rich.console import Console
from ui.virtual_patient import run_interactive_dashboard

console = Console()

def main() -> int:
    console.print("[bold cyan]=== RUNNING WHOLE-BODY COMPUTATIONAL PSYCHIATRY PIPELINE ===[/bold cyan]")
    
    # Run entire pytest suite including go/no-go gates
    exit_code = pytest.main(["-q", "--strict-markers"])
    if exit_code != 0:
        console.print("[bold red]Pipeline halted: Test suite failures detected.[/bold red]")
        return exit_code

    console.print("[bold green]All architectural axioms, mass balances, and validation gates passed successfully![/bold green]\n")
    run_interactive_dashboard()
    return 0

if __name__ == "__main__":
    sys.exit(main())
