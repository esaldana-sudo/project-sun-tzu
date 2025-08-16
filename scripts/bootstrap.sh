#!/usr/bin/env bash
set -euo pipefail

# Resolve absolute path to repo root (this script's parent directory)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[*] Creating project folders..."
mkdir -p "$ROOT/data/replays" "$ROOT/data/processed" "$ROOT/data/maps"
mkdir -p "$ROOT/models" "$ROOT/outputs/heatmaps"

echo "[*] Ensuring .gitkeep files exist..."
touch "$ROOT/data/.gitkeep" "$ROOT/data/maps/.gitkeep" "$ROOT/data/replays/.gitkeep"
touch "$ROOT/data/processed/.gitkeep" "$ROOT/models/.gitkeep"
touch "$ROOT/outputs/.gitkeep" "$ROOT/outputs/heatmaps/.gitkeep"

echo "[*] Setting up virtual environment..."
python3 -m venv "$ROOT/.venv"
source "$ROOT/.venv/bin/activate"

echo "[*] Installing Python dependencies..."
pip install --upgrade pip
pip install -r "$ROOT/requirements.txt"

echo "[✅] Bootstrap complete."
echo "To begin: cd $ROOT && source .venv/bin/activate"

