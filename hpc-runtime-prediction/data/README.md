   # Dataset Information

   ## NREL Eagle Supercomputer Jobs Dataset

   This research uses the publicly available NREL Eagle Supercomputer dataset containing real HPC workload traces.

   ### Dataset Details

   - **Source:** National Renewable Energy Laboratory (NREL)
   - **System:** Eagle Supercomputer
   - **Time Period:** November 2018 - February 2023
   - **Total Jobs:** 11,030,377
   - **URL:** [https://data.openei.org/submissions/5860](https://data.openei.org/submissions/5860)

   ### Download Instructions

   ⚠️ **Important:** The dataset is NOT included in this repository due to its size (115-253 MB).

   #### Option 1: Parquet Format (Recommended)
   ```bash
   # Download eagle_data.parquet (253 MB)
   wget https://data.openei.org/files/5860/eagle_data.parquet

   # Or use browser to download from:
   # https://data.openei.org/submissions/5860
   ```

   #### Option 2: Compressed CSV
   ```bash
   # Download eagle_data.csv.bz2 (115 MB)
   wget https://data.openei.org/files/5860/eagle_data.csv.bz2
   ```

   **Place the downloaded file in the project root directory.**

   ### Dataset Schema

   The Eagle dataset contains the following columns:

   | Column | Description | Type | Example |
   |--------|-------------|------|---------|
   | `job_id` | Unique job identifier | int | 123456 |
   | `user` | User ID (anonymized) | string | user_001 |
   | `account` | Account/project name | string | project_a |
   | `partition` | Queue/partition name | string | standard |
   | `qos` | Quality of Service | string | normal |
   | `wallclock_req` | Requested time limit | string | 01:00:00 |
   | `nodes_req` | Number of nodes requested | int | 4 |
   | `processors_req` | Number of processors | int | 144 |
   | `gpus_req` | Number of GPUs requested | int | 0 |
   | `mem_req` | Memory requested | string | 4000M |
   | `submit_time` | Job submission timestamp | datetime | 2020-01-15 10:30:00 |
   | `start_time` | Job start timestamp | datetime | 2020-01-15 10:35:00 |
   | `end_time` | Job end timestamp | datetime | 2020-01-15 11:20:00 |
   | `run_time` | Actual runtime (seconds) | int | 2700 |
   | `state` | Job completion state | string | COMPLETED |
   | `name` | Job name | string | simulation_001 |
   | `work_dir` | Working directory | string | /projects/... |
   | `submit_line` | SLURM submission command | string | sbatch script.sh |

   ### Data Preprocessing

   Our analysis performs the following preprocessing steps:

   1. **Time Conversion**
      - Convert `run_time` from seconds to hours
      - Parse `wallclock_req` from format "HH:MM:SS" or "D-HH:MM:SS"

   2. **Memory Parsing**
      - Parse `mem_req` from formats: "4G", "4000M", "4000000K"
      - Convert all to MB for consistency

   3. **Feature Engineering**
      - `ntasks`: Estimated from processors and nodes
      - `cpus_per_task`: Calculated from total CPUs
      - `mem_per_cpu`: Memory per CPU in MB

   4. **Data Cleaning**
      - Remove jobs with missing values
      - Remove jobs with zero runtime
      - Remove jobs where runtime > time_limit (data errors)
      - Remove outliers (>99th percentile)

   5. **Sampling**
      - Stratified sampling by partition
      - Final sample: 213,362 jobs (for computational efficiency)

   ### Statistics (After Preprocessing)

   ```
   Total jobs analyzed: 213,362
   Median runtime: 0.175 hours (10.5 minutes)
   Mean runtime: 2.36 hours
   Median time limit: 4.00 hours
   Mean utilization: 18.01%
   Jobs with <60% utilization: 90.8%
   ```

   ### Citation

   If you use this dataset, please cite:

   ```bibtex
   @misc{nrel2023eagle,
   title={NREL Eagle Supercomputer Jobs Dataset},
   author={{National Renewable Energy Laboratory}},
   year={2023},
   publisher={OpenEI},
   howpublished={\url{https://data.openei.org/submissions/5860}},
   note={Accessed: 2025-10-22}
   }
   ```

   ### License

   The NREL Eagle dataset is provided under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

   ### Additional Information

   - **System Specifications:** Eagle is a 7.16 petaFLOPS supercomputer
   - **Purpose:** Renewable energy and computational science research

   ### Support

   For questions about the dataset, contact NREL:
   - **Website:** [https://www.nrel.gov/hpc/eagle-system.html](https://www.nrel.gov/hpc/eagle-system.html)
   - **Data Portal:** [https://data.openei.org](https://data.openei.org)
