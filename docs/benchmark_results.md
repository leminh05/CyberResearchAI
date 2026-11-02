# Benchmark: plain Python vs PySpark (local mode)

- Machine: Windows 10, 12 logical CPUs, Python 3.11.0
- Spark: `local[*]` (single JVM on this machine, **not a cluster**), driver memory 2g
- Spark session startup (paid once): **4.2 s**
- Timing: best of repeats, after one warm-up run (3 repeats)
- Base dataset: 381 real source records from `paper_sources`; larger sizes are synthetic copies with new IDs and ~20% injected duplicates

| Records | File (MB) | Python (s) | Spark (s) | Spark / Python | Python peak heap (MB) | Spark JVM used (MB) | Kept after dedup | Duplicates removed | Same result |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1,000 | 1.74 | 0.083 | 0.866 | 10.43x | 4.0 | 437.1 | 878 | 122 | yes |
| 10,000 | 17.65 | 0.824 | 1.095 | 1.33x | 39.7 | 212.0 | 8,094 | 1,906 | yes |
| 100,000 | 177.0 | 8.989 | 4.168 | 0.46x | 397.1 | 505.0 | 80,580 | 19,420 | yes |
| 300,000 | 531.71 | 26.74 | 8.788 | 0.33x | 1194.9 | 848.6 | 242,098 | 57,902 | yes |
