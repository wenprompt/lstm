# CLAUDE.md

This file provides comprehensive guidance to Claude Code when working with Python code in this repository.

## Core Development Philosophy

### KISS (Keep It Simple, Stupid)

Simplicity should be a key goal in design. Choose straightforward solutions over complex ones whenever possible. Simple solutions are easier to understand, maintain, and debug.

### YAGNI (You Aren't Gonna Need It)

Avoid building functionality on speculation. Implement features only when they are needed, not when you anticipate they might be useful in the future.

### Design Principles

- **Dependency Inversion**: High-level modules should not depend on low-level modules. Both should depend on abstractions.
- **Open/Closed Principle**: Software entities should be open for extension but closed for modification.
- **Single Responsibility**: Each function, class, and module should have one clear purpose.
- **Fail Fast**: Check for potential errors early and raise exceptions immediately when issues occur.

## 🧱 Code Structure & Modularity

### File and Function Limits

- **Never create a file longer than 500 lines of code**. If approaching this limit, refactor by splitting into modules.
- **Functions should be under 50 lines** with a single, clear responsibility.
- **Classes should be under 100 lines** and represent a single concept or entity.
- **Organize code into clearly separated modules**, grouped by feature or responsibility.
- **Line lenght should be max 100 characters** ruff rule in pyproject.toml
- **Use venv_linux** (the virtual environment) whenever executing Python commands, including for unit tests.

### Project Architecture

The LSTM iron ore forecasting project is organized into the following modular structure:

```
src/
├── data/                    # Data loading and preprocessing
│   ├── data_loader.py      # Main data loading with chronological splits
│   └── dataset.py          # PyTorch Dataset and DataLoader creation
├── models/                  # Model architecture definitions
│   └── model.py            # Bidirectional LSTM model implementation
├── training/                # Training pipeline and optimization
│   └── train.py            # Training loop with early stopping
├── evaluation/              # Model evaluation and metrics
│   └── evaluate.py         # Comprehensive evaluation with visualizations
├── features/                # Feature engineering utilities
├── config/                  # Configuration management
└── utils/                   # General utilities

scripts/                     # Data preprocessing scripts
├── build_continuous_futures.py     # M+1 futures series construction
└── build_consolidated_features.py  # 12-feature dataset consolidation

main.py                      # Main orchestration script
config.yaml                  # Centralized hyperparameter configuration
```

## 🛠️ Development Environment

### Development Commands

#### Code Quality and Testing

```bash
# Run all tests
uv run pytest

# Run specific tests with verbose output
uv run pytest tests/test_module.py -v

# Run tests with coverage
uv run pytest --cov=src --cov-report=html

# Format code
uv run ruff format .

# Check linting
uv run ruff check .

# Fix linting issues automatically
uv run ruff check --fix .

# Type checking
uv run mypy src/

# Run pre-commit hooks
uv run pre-commit run --all-files
```

#### Data Processing Pipeline

```bash
# Build continuous M+1 futures series (65% DSP + 62% FEF)
uv run python scripts/build_continuous_futures.py

# Build consolidated 12-feature dataset with Y target
uv run python scripts/build_consolidated_features.py

# Inspect raw pickle files (debugging)
uv run python scripts/inspect_pkl_files.py
```

#### LSTM Model Pipeline

```bash
# Run complete LSTM training and evaluation pipeline
uv run python main.py

# Individual module testing (if needed)
uv run python -c "from src.models.model import create_model; print('Model import successful')"
uv run python -c "from src.data.data_loader import DataLoader; print('DataLoader import successful')"
```

To run the tuning script:

```bash
uv run python -m src.tuning.tune_hyperparameters
```

To run the visualization script (after tuning is complete):

```bash
uv run python -m src.tuning.visualize_tuning
```

### What Happens When You Run `main.py`

When you execute `uv run python main.py`, the complete LSTM pipeline runs automatically:

#### Step-by-Step Execution Flow

1. **Configuration Loading**

   ```
   Loading configuration from config.yaml
   Model: 64 hidden units, 2 layers
   Training: 100 max epochs, batch size 32
   ```

2. **Data Loading & Preprocessing**

   ```
   📊 Data Split: Using 1.7 years for training, 868 total daily observations
   🔧 Features Prepared: All 12 features normalized to 0-1 range for optimal neural network training
   ```

3. **Model Creation**

   ```
   🧠 Neural Network: MODERATE model with 50K parameters, designed to learn iron ore price patterns
   ```

4. **Training Process** (Most Important Phase)

   ```
   Epoch 15/100 - Training Phase
       Batch  19/ 19 | Loss: 1.234567 | Avg Loss: 1.345678 | Progress:  95.0%
   Epoch 15/100 - Validation Phase
   Epoch 15/100 Complete:
     Train Loss: 1.234567
     Val Loss:   1.123456 (↓+0.111111)
     Time:       12.5s
     LR:         0.001000
     Best Val:   1.123456
     📊 Summary: Model performance is IMPROVING, training speed is FAST
   ```

5. **Training Completion**

   ```
   🎯 Training Summary: stopped early (model stopped improving), took moderate time (25.3 min)
   ```

6. **Model Evaluation**

   ```
   📈 Model Quality: GOOD direction prediction (58.5%), LOW error rate, MODERATE correlation
   ```

7. **Final Results**
   ```
   🎯 FINAL RESULT: GOOD iron ore price forecasting model - predicts price direction correctly 58.5% of the time
   ```

#### Expected Results by Dataset

Think of training an LSTM model like teaching someone to predict iron ore prices:

**Training Dataset (607 samples, ~70%) - "The Learning Phase"**

- **What it does**: Like showing a student 607 examples of "when X happened, price went up/down"
- **Purpose**: The model memorizes patterns from March 2022 to ~Jan 2024
- **What you'll see**: Training loss starts high (~3.0) and drops as model learns (~1.5)
- **Good sign**: Loss keeps decreasing = model is learning patterns
- **Bad sign**: Loss stays flat = model can't find patterns in your data

**Validation Dataset (130 samples, ~15%) - "The Practice Test"**

- **What it does**: Like giving the student practice problems they've never seen
- **Purpose**: Check if model learned real patterns (not just memorization)
- **What you'll see**: Validation loss should improve alongside training loss
- **Early stopping**: If validation stops improving but training keeps going = model is memorizing instead of learning real patterns
- **Good sign**: Validation loss improves for 10-20 epochs then plateaus
- **Bad sign**: Validation loss goes up while training loss goes down = overfitting

**Test Dataset (131 samples, ~15%) - "The Final Exam"**

- **What it does**: Final unseen data to see how good the model really is
- **Purpose**: Honest evaluation of real-world performance (~Feb 2024 to Aug 2025)
- **What you'll see**: Final metrics that tell you if the model works in practice
- **Expected Performance**:
  - **50-65% Directional Accuracy**: Can the model predict if price goes up or down better than flipping a coin?
  - **1.0-2.5% RMSE**: How far off are the predictions on average?
  - **0.8-2.0% MAE**: Average size of prediction errors

**Why Split This Way?**

- **Training**: Model needs lots of examples to learn (70%)
- **Validation**: Need enough data to reliably detect overfitting (15%)
- **Test**: Need fresh data for honest performance assessment (15%)
- **Chronological order**: We use oldest data for training, newest for testing (realistic scenario)

#### Output Files & Results Location

**Training Logs**

```
results/training.log          # Complete training history
Console output                # Real-time progress with emojis
```

**Model Checkpoints**

```
results/models/best_model.pt   # Best performing model weights
results/models/final_model.pt  # Final epoch model weights
```

**Evaluation Results**

```
results/evaluation_results.json     # All metrics in JSON format
results/final_results.json          # Complete pipeline results
```

**Visualization Plots**

```
results/plots/actual_vs_predicted_scatter.png    # Overall prediction quality
results/plots/timeseries_comparison.png          # Predictions over time
results/plots/residuals_analysis.png             # Error distribution analysis
results/plots/distribution_comparison.png        # Statistical comparison
```

#### Performance Interpretation Guide

**Layman Summary Meanings**

**Training Performance**:

- **IMPROVING**: ✅ Model is getting better each epoch
- **PLATEAU**: ⚠️ Model stopped improving but stable
- **DECLINING**: ❌ Model is getting worse (overfitting)

**Training Speed**:

- **FAST**: < 10 seconds per epoch (good hardware/small model)
- **MEDIUM**: 10-30 seconds per epoch (normal)
- **SLOW**: > 30 seconds per epoch (large model/slow hardware)

**Model Quality**:

- **EXCELLENT**: ≥60% directional accuracy (very good forecasting)
- **GOOD**: 55-59% directional accuracy (solid performance)
- **FAIR**: 50-54% directional accuracy (slightly better than random)
- **POOR**: <50% directional accuracy (needs improvement)

**Error Assessment**:

- **LOW**: RMSE < 1.0% (highly accurate predictions)
- **MODERATE**: RMSE 1.0-2.0% (acceptable accuracy)
- **HIGH**: RMSE > 2.0% (significant prediction errors)

**Final Success Criteria**

🎯 **Excellent Model** (≥60% directional accuracy):

- Predicts iron ore price direction better than most traders
- Ready for real-world forecasting applications
- Strong commercial value

🎯 **Good Model** (55-59% directional accuracy):

- Solid forecasting performance above random chance
- Useful for trend analysis and risk management
- Good foundation for further improvement

🎯 **Decent Model** (50-54% directional accuracy):

- Basic pattern recognition working
- Slightly better than random guessing
- Needs hyperparameter tuning or more data

🎯 **Needs Work** (<50% directional accuracy):

- Model not learning meaningful patterns
- Check data quality, feature engineering, or model architecture
- May need different approach

## 🧠 LSTM Implementation Guide

### Architecture Overview

The LSTM iron ore forecasting system follows a **modular pipeline architecture**:

```
📊 Raw Data → 🔄 Preprocessing → 🧠 Model → 📈 Evaluation → 💾 Results
```

### Data Flow & File Interactions

#### Phase 1: Data Preprocessing (Run Once)

1. **Continuous Futures Construction** (`scripts/build_continuous_futures.py`)

   - Input: Raw futures contracts (69 months 65% DSP, 81 months 62% FEF)
   - Process: Backward cumulative adjustment method for M+1 series
   - Output: 2 continuous daily price series

2. **Feature Consolidation** (`scripts/build_consolidated_features.py`)
   - Input: 6 data sources with mixed frequencies
   - Process: Date alignment, forward-fill for weekly data, Y target calculation
   - Output: 12 features + Y target (868 samples from March 2022)

#### Phase 2: LSTM Pipeline (`main.py`)

1. **Data Loading** (`src/data/data_loader.py`)

   - Chronological splits: 70%/15%/15% (train/val/test)
   - MinMaxScaler on features only (Y target unscaled)

2. **Dataset Creation** (`src/data/dataset.py`)

   - Sliding window sequences: 20 timesteps → 1 prediction
   - PyTorch DataLoaders with batch_size=32

3. **Model Architecture** (`src/models/model.py`)

   ```
   Input (12 features) → Bidirectional LSTM (96 hidden, 2 layers)
   → LayerNorm → Dropout(0.35) → Linear → Output (1 prediction)
   ```

4. **Training** (`src/training/train.py`)

   - Adam optimizer (lr=0.001) with gradient clipping
   - Early stopping (patience=15) monitoring validation loss
   - Model checkpointing saves best weights

5. **Evaluation** (`src/evaluation/evaluate.py`)
   - Metrics: RMSE, MAE, Directional Accuracy, R², MAPE
   - Visualizations: 4 comprehensive plots saved to results/

### Key Configuration Parameters

Edit `config.yaml` to modify:

- **Model**: `hidden_size: 96`, `sequence_length: 20`, `num_layers: 2`
- **Training**: `learning_rate: 0.001`, `batch_size: 32`, `epochs: 200`
- **Data**: Chronological split ratios, feature scaling settings

### Troubleshooting & Debugging

#### Common Issues & Solutions

- **Import errors**: Ensure proper module structure with `__init__.py` files
- **CUDA errors**: Set `device.use_cuda: false` in config.yaml for CPU-only
- **Memory issues**: Reduce `batch_size` or `sequence_length` in config.yaml
- **Poor performance**: Check feature scaling, try different `hidden_size`

#### Monitoring Training

- **Live progress**: Watch console logs during training
- **Training history**: Check `results/training.log`
- **Model checkpoints**: Best model saved at `results/models/best_model.pt`
- **Evaluation results**: JSON metrics at `results/evaluation_results.json`

#### Performance Analysis Locations

- **Training metrics**: Final console output + training logs
- **Test performance**: `results/evaluation_results.json`
- **Visual analysis**: 4 plots in `results/plots/`
  - `actual_vs_predicted_scatter.png` - Overall prediction quality
  - `timeseries_comparison.png` - Temporal prediction patterns
  - `residuals_analysis.png` - Error distribution analysis
  - `distribution_comparison.png` - Statistical comparison

### File-by-File Responsibilities

| File                         | Purpose                | Key Classes/Functions          | When to Modify               |
| ---------------------------- | ---------------------- | ------------------------------ | ---------------------------- |
| `config.yaml`                | Hyperparameters        | Configuration dict             | Tuning model/training        |
| `src/data/data_loader.py`    | Data preprocessing     | `DataLoader`                   | Changing data splits/scaling |
| `src/data/dataset.py`        | PyTorch datasets       | `LSTMTimeSeriesDataset`        | Modifying sequence creation  |
| `src/models/model.py`        | LSTM architecture      | `IronOreLSTM`                  | Changing model structure     |
| `src/training/train.py`      | Training pipeline      | `LSTMTrainer`, `EarlyStopping` | Training modifications       |
| `src/evaluation/evaluate.py` | Model evaluation       | `ModelEvaluator`               | Adding new metrics/plots     |
| `main.py`                    | Pipeline orchestration | Pipeline functions             | Workflow changes             |

## 📋 Style & Conventions

### Python Style Guide

- **Follow PEP8** with these specific choices:
  - Line length: 100 characters (set by Ruff in pyproject.toml)
  - Use double quotes for strings
  - Use trailing commas in multi-line structures
- **Always use type hints** for function signatures and class attributes
- **Format with `ruff format`** (faster alternative to Black)
- **Use `pydantic` v2** for data validation and settings management

### Docstring Standards

Use Google-style docstrings for all public functions, classes, and modules:

```python
def calculate_discount(
    price: Decimal,
    discount_percent: float,
    min_amount: Decimal = Decimal("0.01")
) -> Decimal:
    """
    Calculate the discounted price for a product.

    Args:
        price: Original price of the product
        discount_percent: Discount percentage (0-100)
        min_amount: Minimum allowed final price

    Returns:
        Final price after applying discount

    Raises:
        ValueError: If discount_percent is not between 0 and 100
        ValueError: If final price would be below min_amount

    Example:
        >>> calculate_discount(Decimal("100"), 20)
        Decimal('80.00')
    """
```

### Naming Conventions

- **Variables and functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private attributes/methods**: `_leading_underscore`
- **Type aliases**: `PascalCase`
- **Enum values**: `UPPER_SNAKE_CASE`

## 🚨 Error Handling

### Exception Best Practices

```python
# Create custom exceptions for your domain
class PaymentError(Exception):
    """Base exception for payment-related errors."""
    pass

class InsufficientFundsError(PaymentError):
    """Raised when account has insufficient funds."""
    def __init__(self, required: Decimal, available: Decimal):
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient funds: required {required}, available {available}"
        )

# Use specific exception handling
try:
    process_payment(amount)
except InsufficientFundsError as e:
    logger.warning(f"Payment failed: {e}")
    return PaymentResult(success=False, reason="insufficient_funds")
except PaymentError as e:
    logger.error(f"Payment error: {e}")
    return PaymentResult(success=False, reason="payment_error")

# Use context managers for resource management
from contextlib import contextmanager

@contextmanager
def database_transaction():
    """Provide a transactional scope for database operations."""
    conn = get_connection()
    trans = conn.begin_transaction()
    try:
        yield conn
        trans.commit()
    except Exception:
        trans.rollback()
        raise
    finally:
        conn.close()
```

### Logging Strategy

```python
import logging
from functools import wraps

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Log function entry/exit for debugging
def log_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Entering {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__} successfully")
            return result
        except Exception as e:
            logger.exception(f"Error in {func.__name__}: {e}")
            raise
    return wrapper
```

## 🔧 Configuration Management

### LSTM Project Configuration (config.yaml)

This project uses a centralized YAML configuration file for all hyperparameters and settings.

#### Threshold Configuration System

The project now uses configurable thresholds for all performance assessments. You can customize these in `config.yaml` under the `thresholds` section:

```yaml
# Training assessment thresholds
thresholds:
  # Performance assessment (validation loss improvement)
  plateau_tolerance: 0.001 # Minimum improvement to avoid "PLATEAU" status

  # Speed assessment (seconds per epoch)
  fast_epoch_time: 10 # Below this = "FAST"
  medium_epoch_time: 30 # Below this = "MEDIUM", above = "SLOW"

  # Total time assessment (seconds)
  quick_training_time: 300 # 5 minutes - Below this = "QUICK"
  moderate_training_time: 1800 # 30 minutes - Below this = "MODERATE", above = "LENGTHY"

  # Model complexity thresholds (parameter count)
  simple_model_params: 10000 # Below this = "SIMPLE"
  moderate_model_params: 100000 # Below this = "MODERATE", above = "COMPLEX"

  # Evaluation quality thresholds
  excellent_accuracy: 60 # Directional accuracy >= this = "EXCELLENT"
  good_accuracy: 55 # Directional accuracy >= this = "GOOD"
  fair_accuracy: 50 # Directional accuracy >= this = "FAIR", below = "POOR"

  strong_correlation: 0.5 # R² >= this = "STRONG"
  moderate_correlation: 0.2 # R² >= this = "MODERATE", below = "WEAK"
```

**How to Tweak Thresholds for Different Outcomes:**

| **Want More...**                   | **Adjust These Thresholds**     | **Example Changes**                           |
| ---------------------------------- | ------------------------------- | --------------------------------------------- |
| **Stricter "EXCELLENT" rating**    | Increase `excellent_accuracy`   | `60 → 65` (need 65% accuracy for "EXCELLENT") |
| **More lenient quality ratings**   | Decrease accuracy thresholds    | `excellent: 60→55, good: 55→50, fair: 50→45`  |
| **Faster training classification** | Increase speed thresholds       | `fast_epoch_time: 10→20` (20s still "FAST")   |
| **Stricter model complexity**      | Decrease parameter thresholds   | `simple_model_params: 10000→5000`             |
| **Higher correlation standards**   | Increase correlation thresholds | `strong_correlation: 0.5→0.7`                 |

**Practical Tuning Examples:**

```yaml
# For iron ore trading (stricter requirements)
thresholds:
  excellent_accuracy: 65  # Need 65% for trading confidence
  good_accuracy: 60      # 60% minimum for trading signals
  fair_accuracy: 55      # 55% minimum for trend analysis

# For research experiments (more lenient)
thresholds:
  excellent_accuracy: 55  # 55% considered excellent for research
  good_accuracy: 52      # 52% considered good
  fair_accuracy: 50      # 50% break-even point
```

#### Feature Selection Configuration

You can also configure which features to use for training/testing by modifying the `features` list in `config.yaml`:

```yaml
# Feature selection - configure which features to use for training/testing
# Available features from consolidated dataset:
features:
  - "price_65_m1" # Continuous M+1 65% iron ore futures price
  - "price_62_m1" # Continuous M+1 62% iron ore futures price
  - "Ukraine Concentrate fines" # Ukraine concentrate fines pricing
  - "lump premium" # Iron ore lump premium
  - "IOCJ Import margin" # IOCJ import margin
  - "rebar steel margin " # Rebar steel margin (note: trailing space in original data)
  - "indian pellet premium" # Indian pellet premium
  - "(IOCJ+SSF)/2-PBF" # Combined index calculation
  - "62 Index" # 62% grade index
  - "65 Index" # 65% grade index
  - "IOCJ Inventory" # IOCJ inventory levels (weekly, forward-filled)
  - "IOCJ Weekly shipment" # IOCJ weekly shipment volumes (weekly, forward-filled)
```

**Usage Examples:**

```yaml
# Use only price-related features (2 features)
features:
  - "price_65_m1"
  - "price_62_m1"

# Use price and index features (4 features)
features:
  - "price_65_m1"
  - "price_62_m1"
  - "62 Index"
  - "65 Index"

# Use single feature for testing (1 feature)
features:
  - "price_65_m1"
```

**Key Benefits:**

- **Experiment with different feature combinations** to find optimal predictive signals
- **Reduce model complexity** by removing less relevant features
- **Focus on domain-specific features** (e.g., only futures prices, only indices)
- **Test feature importance** by comparing models with different feature sets

**Technical Notes:**

- The Y target variable is automatically preserved regardless of feature selection
- Model input size is dynamically calculated based on selected features
- Feature scaling is applied only to selected features
- All selected features must exist in the consolidated dataset

### Environment Variables and Settings

```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with validation."""
    app_name: str = "MyApp"
    debug: bool = False
    database_url: str
    redis_url: str = "redis://localhost:6379"
    api_key: str
    max_connections: int = 100

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

# Usage
settings = get_settings()
```

## 🏗️ Data Models and Validation

### Example Pydantic Models strict with pydantic v2

```python
from pydantic import BaseModel, Field, validator, EmailStr
from datetime import datetime
from typing import Optional, List
from decimal import Decimal

class ProductBase(BaseModel):
    """Base product model with common fields."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    price: Decimal = Field(..., gt=0, decimal_places=2)
    category: str
    tags: List[str] = []

    @validator('price')
    def validate_price(cls, v):
        if v > Decimal('1000000'):
            raise ValueError('Price cannot exceed 1,000,000')
        return v

    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }

class ProductCreate(ProductBase):
    """Model for creating new products."""
    pass

class ProductUpdate(BaseModel):
    """Model for updating products - all fields optional."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    price: Optional[Decimal] = Field(None, gt=0, decimal_places=2)
    category: Optional[str] = None
    tags: Optional[List[str]] = None

class Product(ProductBase):
    """Complete product model with database fields."""
    id: int
    created_at: datetime
    updated_at: datetime
    is_active: bool = True

    class Config:
        from_attributes = True  # Enable ORM mode
```

## 🛡️ Security Best Practices

### Security Guidelines

- Never commit secrets - use environment variables
- Validate all user input with Pydantic
- Use parameterized queries for database operations
- Implement rate limiting for APIs
- Keep dependencies updated with `uv`
- Use HTTPS for all external communications
- Implement proper authentication and authorization

### Example Security Implementation

```python
from passlib.context import CryptContext
import secrets

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def generate_secure_token(length: int = 32) -> str:
    """Generate a cryptographically secure random token."""
    return secrets.token_urlsafe(length)
```

## 🔍 Debugging Tools

### Debugging Commands

```bash
# Interactive debugging with ipdb
uv add --dev ipdb
# Add breakpoint: import ipdb; ipdb.set_trace()

# Memory profiling
uv add --dev memory-profiler
uv run python -m memory_profiler script.py

# Line profiling
uv add --dev line-profiler
# Add @profile decorator to functions

# Debug with rich traceback
uv add --dev rich
# In code: from rich.traceback import install; install()
```

## 📊 Monitoring and Observability

### Structured Logging

```python
import structlog

logger = structlog.get_logger()

# Log with context
logger.info(
    "payment_processed",
    user_id=user.id,
    amount=amount,
    currency="USD",
    processing_time=processing_time
)
```

## 📚 Useful Resources

### Essential Tools

- UV Documentation: https://github.com/astral-sh/uv
- Ruff: https://github.com/astral-sh/ruff
- Pytest: https://docs.pytest.org/
- Pydantic: https://docs.pydantic.dev/
- FastAPI: https://fastapi.tiangolo.com/

### Python Best Practices

- PEP 8: https://pep8.org/
- PEP 484 (Type Hints): https://www.python.org/dev/peps/pep-0484/
- The Hitchhiker's Guide to Python: https://docs.python-guide.org/

## ⚠️ Important Notes

- **NEVER ASSUME OR GUESS** - When in doubt, ask for clarification
- **Always verify file paths and module names** before use
- **Keep CLAUDE.md updated** when adding new patterns or dependencies
- **Test your code** - No feature is complete without tests
- **Document your decisions** - Future developers (including yourself) will thank you

## 🔍 Search Command Requirements

**CRITICAL**: Always use `rg` (ripgrep) instead of traditional `grep` and `find` commands:

```bash
# ❌ Don't use grep
grep -r "pattern" .

# ✅ Use rg instead
rg "pattern"

# ❌ Don't use find with name
find . -name "*.py"

# ✅ Use rg with file filtering
rg --files | rg "\.py$"
# or
rg --files -g "*.py"
```

**Enforcement Rules:**

```
(
    r"^grep\b(?!.*\|)",
    "Use 'rg' (ripgrep) instead of 'grep' for better performance and features",
),
(
    r"^find\s+\S+\s+-name\b",
    "Use 'rg --files | rg pattern' or 'rg --files -g pattern' instead of 'find -name' for better performance",
),
```
