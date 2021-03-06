# sargas - A molecular simulation code for teaching

**This project is work in progress and probably contains bugs**

With **sargas** you can conduct simple molecular simulations of spherical, soft particles from within a jupyter notebook.
It is intended for teaching students the fundamentals of molecular simulations.

## Features

- Monte Carlo Simulations
  - NVT (particle displacement)
  - NPT (particle displacement + volume change)
  - µVT (particle displacement + particle insertion/deletion)
- Molecular Dynamics Simulations
  - NVE (Velocity-Verlet)
  - NVT (thermostats: velocity rescaling, Andersen)
- Chemical potential computation using Widom's insertion method

## Examples

Take a look at the jupyter notebooks in the `examples` folder.

## Installation

The library is written in [Rust](https://www.rust-lang.org/) with bindings for Python 3 via [PyO3](https://github.com/PyO3/pyo3).

### Compiling the Rust library

To build the rust library, use

```terminal
cargo build --release
```

or

```terminal
RUSTFLAGS="-C target-cpu=native" maturin develop --release
```

We found the latter option to drastically improve simulation times.

### Building the Python package

To build the python package, we use [`maturin`](https://github.com/PyO3/maturin), which needs either a `conda` environment or a `virtualenv`.

Here is an example using `virtualenv`, where we name our environment *sargas_venv*.

```terminal
pip install virtualenv
virtualenv sargas_venv
source sargas_venv/bin/activate
pip install numpy pandas ipykernel maturin
```

Then, to build the development version, type

```terminal
RUSTFLAGS="-C target-cpu=native" maturin develop --release
```

or, to build wheels, type

```terminal
RUSTFLAGS="-C target-cpu=native" maturin build --release
```

Finally, it is useful to create an Ipython kernel for our environment for easy use in jupyter lab or notebooks.

```
python -m ipykernel install --user --name sargas --display-name "sargas"
```

### Running tests

To run the tests, deactivate the default features (that is, the python related crates)

```
cargo test --release --no-default-features
```

---

[**sargas**](https://en.wikipedia.org/wiki/Theta_Scorpii) is an evolved bright giant and one of the brightest stars in the night sky.
