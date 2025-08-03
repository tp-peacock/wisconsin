# wisconsin

Preliminary analysis of the Diagnostic Wisconsin Breast Cancer Database

## installation

### Python and Package Management

ASDF was used as a manager for Python versioning and Python package manager versioning for this project.

If you would like you use ASDF, it fast and simple to install. You can get started [here](https://asdf-vm.com/guide/getting-started.html).

Once installed, simply run the following in the root of this repository:

Then run:
```bash
asdf install
```
(This uses the `.tool-versions` file in this repository to install compatible versions of python and the PDM python package manager.)

### Packages

The packages required to run this pipeline are listed in `pyproject.toml` and should be compatible with most common package managers.

During development, [PDM](https://pdm-project.org/en/latest/) was used for package management. If you follow the ASDF instructions above, PDM will be automatically installed. Then, to install dependencies using PDM, run:

```bash
pdm install
```

This command creates a virtual environment in the `.venv` directory. You can activate it with:

```bash
source .venv/bin/activate
```