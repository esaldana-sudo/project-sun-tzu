# 🧠 Project Sun Tzu

**AI Positional Strategy Engine for Overwatch 2**

This project extracts spatial and strategic insights from Overwatch 2 replays. It uses machine learning to identify effective positioning strategies by map and hero class (Tank, DPS, Support).

## 🔧 Pipeline Overview
1. Parse Overwatch replays
2. Map player positions to grid
3. Engineer spatial + temporal features
4. Train ML model to detect “good” positions
5. Visualize strategies using heatmaps

## 📂 Repo Structure

- `src/`: Source code (parser, modeling, visualizations)
- `data/`: Replay inputs, map metadata, outputs
- `models/`: Trained model files
- `notebooks/`: Model evaluation & prototyping
- `outputs/`: Final outputs for review or analysis

## ✅ Status
| Phase | Description                        | Status    |
|-------|------------------------------------|-----------|
| 1     | Replay Parsing                     | 🔧 In Progress |
| 2     | Map Spatial Modeling               | ⏳ Planned |
| 3     | Feature Engineering                | ⏳ Planned |
| 4     | Model Training                     | ⏳ Planned |
| 5     | Visualization & Output             | ⏳ Planned |
| 6     | CLI Inference Tool                 | ⏳ Planned |

## 📜 License
MIT License (or your preferred open source license)
