# Running the GraphSAGE rung on YelpChi and Amazon

The two GraphSAGE cells still marked *pending* in the README. About 10–15
minutes on an NVIDIA GPU (several hours on a laptop CPU). Commands are for
Windows PowerShell, run from the repository folder.

## 1. Install PyTorch with CUDA

Get the exact command for your CUDA version from the selector at
https://pytorch.org/get-started/locally/ (Stable, Windows, Pip, Python, CUDA).
It looks like:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The last line must print `True`. If it prints `False`, the CPU build was
installed: uninstall torch and repeat with the CUDA command.

## 2. Get the data

```powershell
git clone --depth 1 https://github.com/YingtongDou/CARE-GNN ..\CARE-GNN
Expand-Archive ..\CARE-GNN\data\YelpChi.zip -DestinationPath raw
Expand-Archive ..\CARE-GNN\data\Amazon.zip -DestinationPath raw
python examples\care\convert_care.py raw\YelpChi.mat data\yelpchi
python examples\care\convert_care.py raw\Amazon.mat data\amazon
```

`raw\` and `data\` are ignored by git, so none of this gets committed.

## 3. Run the audits

```powershell
graphlift audit data\amazon --gnn --out data\amazon_report
graphlift audit data\yelpchi --gnn --out data\yelpchi_report
```

Each prints one line per seed ending in `gnn 0.xxx`. Three seeds, five folds.

## 4. Update the results

```powershell
copy data\amazon_report\report.md results\amazon.md
copy data\amazon_report\report.json results\amazon.json
copy data\yelpchi_report\report.md results\yelpchi.md
copy data\yelpchi_report\report.json results\yelpchi.json
graphlift index results\synenv_balanced_s2.json results\yelpchi.json results\elliptic_local.json results\elliptic_all.json results\amazon.json --out results\INDEX.md
```

Then replace the four *pending* cells in the README's index table with the
GraphSAGE and GraphSAGE-lift numbers from `results\INDEX.md`.
