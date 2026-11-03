# Regenerating the local analysis plots

Three plot folders are not tracked (`.gitignore`). They are built from the experiment data on
S3 with scripts in this repo:

| folder | script | data |
|---|---|---|
| `plots/assessment_plots/` | `scripts/plotting/plot_assessment_data.py` | Exp2 human-human survey answers |
| `plots/experiment/switching/` | `experiment_analysis/switching_analysis.py` | Exp1 rounds |
| `plots/exp2/` | none: the script is lost | Exp2 pilot `z1awq43i` |

## 1. Get the data

The scripts read the DB exports under their old local names in `data/`. The S3 copies are
byte-identical (bucket `s3://chm-algorithmic-institutions`, see its `README.md`):

| local file | S3 key |
|---|---|
| `data/round.csv` | `experiment_db/raw_exports/rounds_until_2025-06-24.csv` |
| `data/human_member_assessment.csv` | `experiment_db/raw_exports/assessment_member_until_2026-04-26.csv` |
| `data/assessment_manager.csv` | `experiment_db/raw_exports/assessment_manager_until_2026-04-07.csv` |

**From the Claude proxy (command server).** `local_path` must lie under
`/opt/algorithmic-institutions` or `/var/cluster_transfer`; a file key downloads to a file path:

```bash
fetch() {  # fetch <local path in the repo> <S3 key>
  run=$(curl -s -X POST http://cc_command_server:8765/s3_sync_from \
    -H 'Content-Type: application/json' \
    -d "{\"local_path\": \"/opt/algorithmic-institutions/$1\", \"key\": \"$2\"}" \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["run_id"])')
  until curl -s "http://cc_command_server:8765/runs/$run" | grep -q '"exited"'; do sleep 2; done
  curl -s "http://cc_command_server:8765/runs/$run" | python3 -c 'import json, sys; print(json.load(sys.stdin)["exit_code"])'
}
fetch data/round.csv experiment_db/raw_exports/rounds_until_2025-06-24.csv
fetch data/human_member_assessment.csv experiment_db/raw_exports/assessment_member_until_2026-04-26.csv
fetch data/assessment_manager.csv experiment_db/raw_exports/assessment_manager_until_2026-04-07.csv
```

Each call prints the exit code of the download (0 is success). The log is
`/var/cc_output/<run_id>.log`.

**With s3cmd directly** (a machine with `~/.s3cfg` for the bucket), from the repo root:

```bash
B=s3://chm-algorithmic-institutions/experiment_db/raw_exports
s3cmd get --force "$B/rounds_until_2025-06-24.csv" data/round.csv
s3cmd get --force "$B/assessment_member_until_2026-04-26.csv" data/human_member_assessment.csv
s3cmd get --force "$B/assessment_manager_until_2026-04-07.csv" data/assessment_manager.csv
```

Check against `MANIFEST.tsv` on the bucket (`s3cmd get s3://chm-algorithmic-institutions/MANIFEST.tsv`):
`sha256sum data/round.csv` must match the row of its key.

## 2. Make the plots

Python deps beyond `pyproject.toml`: `pip install fire ykutil` (the switching script needs both).

```bash
# survey plots: the 6 Exp2 human-human dataset sessions (a few seconds)
python scripts/plotting/plot_assessment_data.py \
  --sessions d6bumnhc aqo5mt8g 8mzxhbhg rokqh2fp yvv20cs6 jkmmr61z

# switching model on the 13 Exp1 sessions (util.prepare_df filters to them)
python experiment_analysis/switching_analysis.py create_switching_plots \
  data/round.csv plots/experiment/switching/first_switch           # about 6 minutes
python experiment_analysis/switching_analysis.py create_whole_game_switching_plot \
  data/round.csv plots/experiment/switching/whole_game_switch      # 24 figures, roughly an hour or more (estimated from first_switch, not measured)
```

## What comes back

- **`plots/assessment_plots/`:** pixel-identical to the plots of 2026-04-29 (checked
  2026-11-03). Without `--sessions` the script uses every session in the export, test runs
  included.
- **`plots/experiment/switching/`:** the same curves up to simulation noise, but not the same
  figures. The originals (2025-07-05) came from an uncommitted variant of the script that
  drew confidence bands and also ran `beta_4`/`beta_5` = 3. The committed script draws no
  bands and runs `beta_4`, `beta_5` in {0, 10}. `first_switch/` was regenerated this way
  (2026-11-03); `whole_game_switch/` was not run to the end.
- **`plots/exp2/`:** no script in either repo makes these five figures (contribution vs
  punishment, contribution variance per group; Exp2 pilot `z1awq43i`, 2025-09-30).

The original figures that cannot be rebuilt are on S3:
`experiment_1/analysis_plots/switching/whole_game_switch/` and
`experiment_2/pilot_ai_governor/analysis_plots/` (the `plots/exp2/` figures). Fetch them like
the data, for example
`s3cmd sync s3://chm-algorithmic-institutions/experiment_2/pilot_ai_governor/analysis_plots/ plots/exp2/`
or `{"local_path": "/opt/algorithmic-institutions/plots/exp2/", "key": "experiment_2/pilot_ai_governor/analysis_plots/"}`.
