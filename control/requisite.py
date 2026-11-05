import pathlib
import yaml

REG = pathlib.Path(__file__).with_name("registry.yaml")


def _load() -> dict:
    with REG.open() as f:
        return yaml.safe_load(f)


def variety() -> tuple[int, int]:
    reg = _load()
    return len(reg["control_dimensions"]), len(reg["perturbations"])


def assert_requisite() -> None:
    r, e = variety()
    if r < e:
        raise AssertionError(
            f"A2 violated: V(R)={r} < V(E)={e}. "
            f"Add {e - r} control dimension(s) before fitting."
        )
