# Benchmark: plain Python vs PySpark (local mode)

- Machine: Windows 10, 12 logical CPUs, Python 3.11.0
- Spark: `local[*]` (single JVM on this machine, **not a cluster**), driver memory 2g
- Spark session startup (paid once): **5.28 s**
- Timing: best of repeats, after one warm-up run (3 repeats)
- Base dataset: 381 real source records from `paper_sources`; larger sizes are synthetic copies with new IDs and ~20% injected duplicates

| Records | File (MB) | Python (s) | Spark (s) | Spark / Python | Python peak heap (MB) | Spark JVM used (MB) | Kept after dedup | Duplicates removed | Same result |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1,000 | 1.74 | 0.086 | 0.902 | 10.49x | 4.0 | 279.9 | 878 | 122 | yes |
| 10,000 | 17.65 | 0.883 | 1.126 | 1.28x | 39.7 | 355.5 | 8,094 | 1,906 | yes |
| 100,000 | 177.0 | 9.027 | 3.292 | 0.36x | 397.1 | 516.2 | 80,580 | 19,420 | yes |
| 300,000 | 531.71 | 27.13 | 8.746 | 0.32x | 1194.9 | 362.1 | 242,098 | 57,902 | yes |
