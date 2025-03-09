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

## Python Environment Installation

### Miniconda

- Make sure miniconda is installed and added to path
- Run ``setup_env.bat`` in your terminal 

Uses `environment.yml`.

### Poetry

- Install poetry ([link](https://python-poetry.org/docs/#installing-with-the-official-installer))
- Run `poetry install` inside `./snntorch/`.

Uses `pyproject.toml`.