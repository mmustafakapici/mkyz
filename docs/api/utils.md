# API Reference - Utils Module

The utils module provides logging and parallel processing utilities.

```python
from mkyz import setup_logging, get_logger
from mkyz.utils import parallel_map, chunk_data
```

---

## Logging

### setup_logging

```python
mkyz.setup_logging(
    level='INFO',
    use_colors=True,
    log_file=None
) -> logging.Logger
```

Configure MKYZ logging.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level` | str | 'INFO' | Log level: DEBUG, INFO, WARNING, ERROR |
| `use_colors` | bool | True | Colored terminal output |
| `log_file` | str | None | Optional file path for logs |

**Example:**
```python
# Basic setup
logger = mkyz.setup_logging(level='DEBUG')
logger.info("Training started")

# With file logging
logger = mkyz.setup_logging(
    level='INFO',
    log_file='logs/training.log'
)
```

**Output:**
```
[14:30:25] [INFO] Training started
[14:30:26] [DEBUG] Loading 10000 samples...
```

---

### get_logger

```python
mkyz.get_logger() -> logging.Logger
```

Get the MKYZ logger instance.

**Example:**
```python
logger = mkyz.get_logger()
logger.info("Processing complete")
logger.warning("Low accuracy detected")
logger.error("Training failed")
```

---

### Log Levels

| Level | Value | Use Case |
|-------|-------|----------|
| `DEBUG` | 10 | Detailed debugging info |
| `INFO` | 20 | General information |
| `WARNING` | 30 | Warning messages |
| `ERROR` | 40 | Error messages |
| `CRITICAL` | 50 | Critical failures |

---

## Parallel Processing

### parallel_map

```python
from mkyz.utils import parallel_map

parallel_map(
    func,
    items,
    n_jobs=-1,
    use_threads=True,
    show_progress=False
) -> list
```

Apply function to items in parallel.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | callable | - | Function to apply |
| `items` | list | - | Items to process |
| `n_jobs` | int | -1 | Workers (-1 = all CPUs) |
| `use_threads` | bool | True | Use threads (vs processes) |
| `show_progress` | bool | False | Show progress bar |

**Returns:** List of results in same order as input

**Example:**
```python
from mkyz.utils import parallel_map

def process_file(filepath):
    df = pd.read_csv(filepath)
    return df.describe()

files = ['data1.csv', 'data2.csv', 'data3.csv', 'data4.csv']
results = parallel_map(process_file, files, n_jobs=4)

# With progress bar
results = parallel_map(
    process_file, 
    files, 
    n_jobs=-1, 
    show_progress=True
)
```

---

### chunk_data

```python
from mkyz.utils import chunk_data

chunk_data(
    data,
    chunk_size=None,
    n_chunks=None
) -> Iterator[list]
```

Split data into chunks for batch processing.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | list | Data to chunk |
| `chunk_size` | int | Size of each chunk |
| `n_chunks` | int | Number of chunks (alternative) |

**Example:**
```python
from mkyz.utils import chunk_data

data = list(range(100))

# By chunk size
for chunk in chunk_data(data, chunk_size=25):
    print(len(chunk))  # 25, 25, 25, 25

# By number of chunks
for chunk in chunk_data(data, n_chunks=4):
    process_batch(chunk)
```

---

### get_optimal_workers

```python
from mkyz.utils.parallel import get_optimal_workers

get_optimal_workers(
    data_size,
    min_items_per_worker=100
) -> int
```

Calculate optimal number of workers.

**Example:**
```python
n_workers = get_optimal_workers(len(data))
results = parallel_map(func, data, n_jobs=n_workers)
```

---

## See Also

- [Quick Start](../quickstart.md)
- [Configuration](../api/core.md)
