# Orbit Simulation Toolkit

Simulates the motion of a planet around a black hole with both classical and relativistic dynamics, using various numerical integration methods.

## Installation

Clone the repository and install in editable mode:

```bash
pip install -e .
```

All required dependencies will be installed automatically.

---

## Usage

### Command Line

You can run the main analyses directly from the command line:

```bash
python -m orbits --mass_bh 5e6 --sm_axis 1. --compare_rel_class --save_gif
```

#### Other examples

1. **Eccentricity Comparison**:

   ```bash
   python -m orbits --mass_bh 5e6 --compare_ecc --save_gif
   ```

2. **Full Analysis**:

   ```bash
   python -m orbits --mass_bh 1e6 --compare_rel_class --compare_methods --convergence
   ```

#### Available Flags

| Flag                  | Description                          | Outputs                      |
|-----------------------|--------------------------------------|------------------------------|
| `--compare_rel_class` | Relativistic vs classical comparison | `rel_vs_class_analysis.csv`  |
| `--compare_methods`   | Integration method comparison        | `methods_comp.csv`           |
| `--convergence`       | Convergence analysis                 | `convergence_results.csv`    |
| `--compare_ecc`       | Eccentricity comparison              | `eccentricity_analysis.csv`  |
| `--save_gif`          | Save animation as GIF                | See `outputfolder/animations`|

You can combine flags as needed.

### In Python

You can also use the toolkit in your own scripts or Jupyter notebooks:

```python
from orbits import TwoBodyProblem, SimulationRunner, AnalysisTools, AnimationCreator

system = TwoBodyProblem(ecc=0.1, mass_bh=5e6, sm_axis=1.0, orb_period=2, method='RK3')
runner = SimulationRunner(system)
t, s = runner.run_simulation()
```

---

## Output Structure

All results and animations are saved in the `outputfolder/` directory:

```bash
outputfolder/
├── orbits_data/       # CSV orbital histories
├── animations/        # GIF animations
└── grids/             # Precomputed grids
```

---

## Project Structure

```bash
orbit-simulation-toolkit
├── orbits/
│   ├── __init__.py
│   └── orbits.py
└── tests/
    ├── tests-orbits.py
    └── TBA
├── analysis.ipynb
├── LICENSE
├── pyproject.toml
├── README.md
└── outputfolder/
    ├── animations/
    ├── grids/
    └── orbits_data/
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Authors

- Wladimir Banda
- Milagros Delgado
