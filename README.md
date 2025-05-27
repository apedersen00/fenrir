# FENRIR

Fast Event-Driven Neural Recognition for dvs InfeRence.

## Branch Conventions

Three main branch groups:

- `docs`: documentation
- `py`: python
- `hw`: HDL and FPGA related work

Subgroups (at the time of writing):

- `8x6`: initial one layer network
- `fenrir`: work on main CNN/SNN hardware 
- `vortex`: python framework for testing and validation

## Upgrading HW Platform in Vitis

1. Generate bitstream in Vivado and `export hardware`.

2. Change the XSA and select the same one again.

3. Build `platform`.

4. Run `update_platform.bat`.

## Python Environment Installation

### Poetry

- Install poetry ([link](https://python-poetry.org/docs/#installing-with-the-official-installer))
- Run `poetry install` inside `./snntorch/`.

Uses `pyproject.toml`.