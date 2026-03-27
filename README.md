# Irrigation Detection

A deep learning pipeline for detecting outdoor irrigation usage from smart water meter (AMI) data. Given hourly consumption readings, the model separates total water usage into irrigation and non-irrigation components.

## How It Works

The pipeline uses a **DilatedUNet1D** — a 1D U-Net with dilated convolutions in the bottleneck for long-range temporal context. The model takes 14-day (336-hour) windows of hourly consumption and directly predicts the irrigation component at each hour.

**Input:** Hourly water meter readings (meter ID, timestamp, reading value)
**Output:** Per-hour decomposition into total, irrigation, and non-irrigation readings

## Quick Start

### From source

```bash
pip install .
irrigation-detect \
    --input data/readings.csv \
    --model models/best_irrigation_model_336h.pth \
    --output results.csv
```

### With Docker

```bash
docker build -t irrigation-detection .
docker run --rm \
    -v $(pwd)/data:/app/data:ro \
    -v $(pwd)/models:/app/models:ro \
    -v $(pwd)/output:/app/output \
    irrigation-detection \
    --input /app/data/readings.csv \
    --model /app/models/best_irrigation_model_336h.pth \
    --output /app/output/results.csv
```

### With Docker Compose

```bash
# Place your input CSV in ./data/ and model in ./models/
docker compose run --rm detect
```

## Input Format

A CSV (or Parquet) file with three columns:

| Column      | Type     | Description                          |
|-------------|----------|--------------------------------------|
| `meter_id`  | string   | Unique identifier for the meter      |
| `timestamp` | datetime | Hourly reading timestamp (UTC)       |
| `reading`   | float    | Consumption value for that hour      |

Example:

```csv
meter_id,timestamp,reading
M001,2024-01-01 00:00:00,0.12
M001,2024-01-01 01:00:00,0.08
M001,2024-01-01 02:00:00,0.05
```

Readings should be **hourly interval values** (consumption during that hour), not cumulative register readings. Values should be non-negative.

## Output Format

| Column                   | Type   | Description                              |
|--------------------------|--------|------------------------------------------|
| `meter_id`               | string | Meter identifier                         |
| `timestamp`              | string | ISO-formatted timestamp                  |
| `total_reading`          | float  | Original consumption value               |
| `irrigation_reading`     | float  | Predicted irrigation component           |
| `non_irrigation_reading` | float  | Predicted non-irrigation component       |

## Model

The DilatedUNet1D architecture:

- **Encoder:** 4 blocks of dilated convolutions with max pooling (1 → 16 → 32 → 64 → 128 channels)
- **Bottleneck:** Dilated convolutions (dilation 2, 4) for multi-scale temporal context (256 channels)
- **Decoder:** 4 upsampling blocks with skip connections
- **Output:** ReLU activation (irrigation is non-negative)
- **Window:** 336 hours (14 days)

The model was trained on synthetic mixed-use meter data. You must supply your own trained model weights (`.pth` file).

## Configuration

All settings are configurable via CLI flags or environment variables:

| CLI Flag       | Env Var        | Default | Description                        |
|----------------|----------------|---------|------------------------------------|
| `--input`      | `INPUT_PATH`   | —       | Path to input CSV/Parquet          |
| `--model`      | `MODEL_PATH`   | —       | Path to model weights (.pth)       |
| `--output`     | `OUTPUT_PATH`  | —       | Path for output CSV                |
| `--device`     | `DEVICE`       | `cpu`   | Torch device (`cpu` or `cuda`)     |
| `--batch-size` | `GPU_BATCH`    | `2000`  | Windows per inference batch        |
| `--source`     | `SOURCE_LABEL` | —       | Optional label for source column   |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run on example data
irrigation-detect --input examples/sample_data.csv --model path/to/model.pth --output out.csv
```

## Citation

If you use this tool in your research, please cite:

```
California Data Collaborative (2026). Irrigation Detection Pipeline.
https://github.com/California-Data-Collaborative/irrigation-detection
```

## License

MIT License. See [LICENSE](LICENSE) for details.
