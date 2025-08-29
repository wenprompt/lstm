# Project Requirements Document (PRD): Iron Ore Log Return Forecasting with LSTM

## 1.0 Project Objective

Develop a **robust and modular Python project** that utilizes an LSTM neural network to forecast the **next-day percentage log return** of the 65% M+1 DSP iron ore price.

**Scope**: Complete ML lifecycle including data preprocessing, feature engineering, model training, evaluation, and visualization.

## 2.0 Background

**Problem**: Forecasting commodity price movements is complex due to:

- Various influencing factors with mixed reporting frequencies (daily, weekly)
- Need for careful handling of futures contract rollovers
- Mixed-frequency data challenges

**Solution**: LSTM model selected for its ability to capture temporal dependencies in time-series data.

## 3.0 Project Structure & Setup

**Environment Manager**: `uv` for virtual environment and package management

```
m65lstm/
├── pyproject.toml              # Project metadata and dependencies for uv
├── data/                       # Input data files
│   ├── group.csv               # 6 daily features
│   ├── IOCJ Inventory.csv      # 1 weekly feature
│   ├── IOCJ Weekly Shipment.csv # 1 weekly feature
│   ├── Raw_65and62_Index.csv   # 2 daily index features
│   ├── Raw_M65F_DSP.pkl        # Raw 65% futures contracts (69 monthly periods)
│   └── Raw_FEF_Close.pkl       # Raw 62% FEF futures contracts (81 monthly periods)
├── notebooks/                  # Jupyter notebooks for exploration and analysis
├── results/                    # Saved outputs (models, plots, metrics)
│   ├── plots/
│   └── metrics.json
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── data_loader.py          # Load and clean 6 data sources
│   ├── features.py             # M+1 futures processing & feature engineering
│   ├── dataset.py              # PyTorch/TensorFlow dataset and sequence creation
│   ├── model.py                # LSTM model architecture
│   ├── train.py                # Training, validation, and early stopping logic
│   └── evaluate.py             # Evaluation and visualization functions
├── config.yaml                 # All hyperparameters and configuration settings
└── main.py                     # Main executable script to run the entire pipeline
```

## 4.0 Input Data

The 12 features are sourced from 6 separate data files requiring different processing approaches:

| File                            | Features      | Frequency | Description                                                                                                                |
| ------------------------------- | ------------- | --------- | -------------------------------------------------------------------------------------------------------------------------- |
| `data/group.csv`                | 6             | Daily     | Ukraine Concentrate fines, lump premium, IOCJ Import margin, rebar steel margin, indian pellet premium\*, (IOCJ+SSF)/2-PBF |
| `data/IOCJ Inventory.csv`       | 1             | Weekly    | IOCJ Inventory levels (requires forward-fill to daily frequency)                                                           |
| `data/IOCJ Weekly Shipment.csv` | 1             | Weekly    | IOCJ Weekly shipment volumes (requires forward-fill to daily frequency)                                                    |
| `data/Raw_65and62_Index.csv`    | 2             | Daily     | 62 Index, 65 Index price indicators                                                                                        |
| `data/Raw_M65F_DSP.pkl`         | 1 (processed) | Raw       | 65% iron ore futures contracts - 69 monthly periods (2020-01 to 2025-09) with daily settlement prices                      |
| `data/Raw_FEF_Close.pkl`        | 1 (processed) | Raw       | 62% FEF iron ore futures contracts - 81 monthly periods (2020-01 to 2026-09) with daily close prices                       |

**Total Features**: 12 (6 daily + 2 forward-filled weekly + 2 daily indices + 2 processed continuous futures series)

**⚠️ Data Availability Constraint**: Indian pellet premium data only available from **March 30, 2022** onwards. All model training and evaluation must use data from this date forward to ensure complete feature coverage.

## 5.0 Functional Requirements

### FR1: Data Loading & Initial Cleaning (`data_loader.py`)

**Inputs**:

- `data/group.csv` (6 daily features)
- `data/IOCJ Inventory.csv` (1 weekly feature)
- `data/IOCJ Weekly Shipment.csv` (1 weekly feature)
- `data/Raw_65and62_Index.csv` (2 daily features)
- `data/Raw_M65F_DSP.pkl` (raw 65% futures data)
- `data/Raw_FEF_Close.pkl` (raw 62% FEF futures data)

**Processing**:

- **CSV Files**: Load using pandas.read_csv() with proper date parsing and encoding handling
- **Pickle Files**: Load dictionaries containing pandas Series indexed by pandas Period objects
- **Weekly Features**: IOCJ Inventory + IOCJ Weekly Shipment
  - Apply **forward-fill methodology** (pandas.fillna(method='ffill')) to convert weekly → daily frequency
  - Handle missing values and ensure proper date alignment
- **Data Validation**: Check for missing dates, data quality issues, and consistent time series coverage
- **Date Filtering**: Filter all datasets to **March 30, 2022 onwards** due to indian pellet premium data availability
- **Output**: Clean, standardized DataFrames ready for feature engineering (covering ~2.4 years of data)

### FR2: M+1 Continuous Futures Series Construction (`features.py`)

**Objective**: Create continuous daily price series for 65% M+1 DSP and 62% M+1 DSP from raw monthly futures contracts.

**Input Sources**:

- `Raw_M65F_DSP.pkl` → **65% M+1 DSP** continuous series
- `Raw_FEF_Close.pkl` → **62% M+1 DSP** continuous series

**M+1 Contract Logic**:

- For any date in month **M**, use the futures contract expiring in month **M+2**
- Example: January 2024 → Use March 2024 contract (M+1)

**Rollover Schedule**: **End-of-Month Transitions**

- **January 31 → February 1**: Switch from March contract to April contract
- **February 29 → March 1**: Switch from April contract to May contract
- **March 31 → April 1**: Switch from May contract to June contract
- Continue monthly pattern...

**Proportional Adjustment Method**:

```python
# At month-end rollover (e.g., Jan 31 → Feb 1):
adjustment_ratio = old_contract_price[last_trading_day] / new_contract_price[last_trading_day]

# Apply to all historical prices of new contract:
adjusted_new_contract = original_new_contract * adjustment_ratio
```

**Algorithm**:

1. **Initialize**: Start with March 2020 contract for January 2020 data
2. **Monthly Rollover**: At each month-end, switch to next M+1 contract
3. **Apply Adjustment**: Use proportional adjustment to maintain price continuity
4. **Chain Series**: Combine adjusted contract segments into continuous daily series
5. **Handle Missing Values**: Interpolate NaN values in final continuous series

**Output**: Two continuous daily price series (65% M+1 DSP, 62% M+1 DSP) preserving historical percentage returns.

### FR3: Feature Engineering & Scaling (`features.py`)

#### Final Feature Set (12 Features Total)

- **6 from `group.csv`** (daily): Ukraine Concentrate fines, lump premium, IOCJ Import margin, rebar steel margin, indian pellet premium, (IOCJ+SSF)/2-PBF
- **2 from weekly sources** (forward-filled to daily): IOCJ Inventory, IOCJ Weekly Shipment
- **2 from `Raw_65and62_Index.csv`** (daily): 62 Index, 65 Index
- **2 from processed futures** (continuous series): 65% M+1 DSP, 62% M+1 DSP

#### Target Variable (y)

```python
# Next-day percentage log return of 65% M+1 DSP:
# For sequence ending on day T, target is:
y = log(price_65_M1_{T+1} / price_65_M1_T) * 100
```

**Source**: 65% M+1 DSP continuous series from processed Raw_M65F_DSP.pkl

#### Input Features (X)

- **Final set**: All 12 processed features aligned to daily frequency
- **Date Alignment**: Ensure all features share consistent daily date index
- **Missing Value Handling**: Forward-fill weekly features, interpolate futures series gaps

#### Scaling Strategy

- ✅ **Input Features**: Apply MinMaxScaler to all 12 features
- ❌ **Target Variable**: Do NOT scale (percentage returns already on suitable scale for neural networks)

### FR4: Model Architecture (`model.py`)

**Architecture**: Stacked, bidirectional LSTM
**Framework**: TensorFlow/Keras or PyTorch

**Configurable Parameters** (via `config.yaml`):

- `number_of_layers`
- `hidden_size`
- `dropout_rate`
- `bidirectional`
- `LayerNorm`

### FR5: Training & Validation Pipeline (`train.py`)

#### Data Splitting (Chronological)

**Data Period**: March 30, 2022 - Present (~2.4 years)

- **Train**: 70% (~1.7 years)
- **Validation**: 15% (~4 months)
- **Test**: 15% (~4 months)

#### Training Configuration

- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Gradient Clipping**: Norm value 1.0 (prevent exploding gradients)

#### Early Stopping

- **Monitor**: Validation loss (`val_loss`)
- **Condition**: Stop if no improvement for `patience` epochs
- **Action**: Restore best model weights

### FR6: Evaluation & Visualization (`evaluate.py`)

#### Metrics (Test Set)

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Directional Accuracy

#### Visualization

- **Plot**: Actual vs. Predicted log returns (entire test set)
- **Save**: `results/plots/`

### FR7: Main Executable & Configuration

#### `main.py`

Orchestrate entire pipeline:

1. Load data
2. Process features
3. Train model
4. Evaluate results

#### `config.yaml`

Centralized configuration for:

- File paths
- Split ratios
- Model hyperparameters
- Training settings

## 6.0 Technical Specifications

### Environment

- **Python**: 3.10+
- **Package Manager**: `uv`

### Core Libraries

- **Data**: Pandas, NumPy
- **ML**: Scikit-learn, TensorFlow 2.x or PyTorch 2.x
- **Visualization**: Matplotlib
- **Config**: PyYAML

### Hyperparameter Space

```yaml
# Hyperparameters for config.yaml
sequence_length: [10, 20]
batch_size: [32, 64]
hidden_size: [64, 96, 128]
number_of_layers: [1, 2]
dropout_rate: [0.2, 0.35, 0.5]
learning_rate: 0.001
bidirectional: true
LayerNorm: true
epochs: 200 # Maximum (controlled by early stopping)
early_stopping_patience: 15
```

## 7.0 Deliverables

### ✅ Required Outputs

1. **Modular Python Project** (structure per Section 3.0)
2. **`pyproject.toml`** - UV dependency management
3. **`config.yaml`** - Hyperparameter management
4. **Evaluation Results**:
   - `results/metrics.json` - Performance metrics
   - `results/plots/` - Visualizations
5. **Well-commented Code** - Clear explanations of key logic

### Success Criteria

- [ ] Complete pipeline runs end-to-end
- [ ] Model trains with early stopping
- [ ] Evaluation metrics generated and saved
- [ ] Code follows modular architecture
- [ ] All parameters configurable via YAML
