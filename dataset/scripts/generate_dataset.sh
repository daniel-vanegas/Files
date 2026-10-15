#!/usr/bin/env bash
set -Eeuo pipefail
# Generate validated map-level route pools. No ROS/Webots launch; no deletion by default.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
DATA_ROOT="$DATASET_DIR/data"
EXPECTED_DATA_ROOT="$REPO_ROOT/src/Navigation/mapless_navigation/dataset/data"
MAPS_ROOT="$REPO_ROOT/src/Hardware/justina_description/webots_env/environments/maps"
GENERATOR="$SCRIPT_DIR/path_generator.py"
N_GOALS="${N_GOALS:-200}"
CLASS_QUOTA="${ST_SEM_CMP_VC:-}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-5}"
SEED_BASE="${SEED_BASE:-2026101400}"
SCENARIOS_TEXT="${SCENARIOS:-E01 E02 E03}"
MODE="goals"; [[ -n "$CLASS_QUOTA" ]] && MODE="quota"
PURGE_DATA=0; ASSUME_YES=0; CLI_TARGET_MODE=""
usage() {
cat <<EOF
Usage: $0 [--n-goals N | --st_sem_cmp_vc S_SC_C_V] [options]
  --n-goals N              balanced total routes per scenario (default: 200)
  --st_sem_cmp_vc QUOTAS   explicit class quotas, e.g. 10_10_15_20
  --balance-class          explicitly select balanced distribution
  --timeout-seconds SEC    optional wall-time limit (0 disables)
  --max-attempts N         optional global attempt cap; omitted means unlimited
  --progress-interval-seconds SEC
  --scenarios "E01 E02"    scenario list (default E01 E02 E03)
  --seed-base N            deterministic seed base
  --purge-data             delete the entire data directory first
  --yes                    skip purge confirmation only with --purge-data
No time/attempt cutoff or data deletion is active unless explicitly requested.
EOF
}
while (($#)); do
 case "$1" in
  --n-goals|--num-paths) (($#>=2))||{ echo "ERROR: $1 needs a value" >&2; exit 2; }; [[ "$CLI_TARGET_MODE" != "quota" ]]||{ echo "ERROR: do not combine --n-goals with --st_sem_cmp_vc" >&2; exit 2; }; CLI_TARGET_MODE="goals"; N_GOALS="$2"; CLASS_QUOTA=""; MODE="goals"; shift 2;;
  --st_sem_cmp_vc|--st-sem-cmp-vc) (($#>=2))||{ echo "ERROR: $1 needs a value" >&2; exit 2; }; [[ "$CLI_TARGET_MODE" != "goals" ]]||{ echo "ERROR: do not combine --st_sem_cmp_vc with --n-goals/--balance-class" >&2; exit 2; }; CLI_TARGET_MODE="quota"; CLASS_QUOTA="$2"; MODE="quota"; shift 2;;
  --balance-class) [[ "$CLI_TARGET_MODE" != "quota" ]]||{ echo "ERROR: do not combine --balance-class with --st_sem_cmp_vc" >&2; exit 2; }; CLI_TARGET_MODE="goals"; CLASS_QUOTA=""; MODE="goals"; shift;;
  --timeout-seconds) (($#>=2))||{ echo "ERROR: $1 needs a value" >&2; exit 2; }; TIMEOUT_SECONDS="$2"; shift 2;;
  --max-attempts) (($#>=2))||{ echo "ERROR: $1 needs a value" >&2; exit 2; }; MAX_ATTEMPTS="$2"; shift 2;;
  --progress-interval-seconds) (($#>=2))||{ echo "ERROR: $1 needs a value" >&2; exit 2; }; PROGRESS_INTERVAL_SECONDS="$2"; shift 2;;
  --scenarios) (($#>=2))||{ echo "ERROR: $1 needs a value" >&2; exit 2; }; SCENARIOS_TEXT="$2"; shift 2;;
  --seed-base) (($#>=2))||{ echo "ERROR: $1 needs a value" >&2; exit 2; }; SEED_BASE="$2"; shift 2;;
  --purge-data) PURGE_DATA=1; shift;;
  --yes) ASSUME_YES=1; shift;;
  -h|--help) usage; exit 0;;
  *) echo "Unknown option: $1" >&2; usage; exit 2;;
 esac
done
if [[ "$MODE" == "goals" ]]; then
 [[ "$N_GOALS" =~ ^[0-9]+$ ]] && ((N_GOALS>=4)) || { echo "ERROR: --n-goals must be an integer >= 4" >&2; exit 2; }
else
 IFS="_" read -r -a QUOTAS <<< "$CLASS_QUOTA"
 (("${#QUOTAS[@]}"==4)) || { echo "ERROR: --st_sem_cmp_vc needs four quotas" >&2; exit 2; }
 for q in "${QUOTAS[@]}"; do [[ "$q" =~ ^[0-9]+$ ]] && ((q>=1)) || { echo "ERROR: quotas must be positive integers" >&2; exit 2; }; done
fi
if [[ -n "$MAX_ATTEMPTS" ]]; then [[ "$MAX_ATTEMPTS" =~ ^[0-9]+$ ]] && ((MAX_ATTEMPTS>=1)) || { echo "ERROR: --max-attempts must be positive" >&2; exit 2; }; fi
if [[ -n "$TIMEOUT_SECONDS" ]]; then
 python3 - "$TIMEOUT_SECONDS" <<'PY' || { echo "ERROR: timeout must be non-negative" >&2; exit 2; }
import sys
if float(sys.argv[1])<0: raise SystemExit(1)
PY
fi
python3 - "$PROGRESS_INTERVAL_SECONDS" <<'PY' || { echo "ERROR: progress interval must be positive" >&2; exit 2; }
import sys
if float(sys.argv[1])<=0: raise SystemExit(1)
PY
[[ "$SEED_BASE" =~ ^[0-9]+$ ]] || { echo "ERROR: seed-base must be non-negative integer" >&2; exit 2; }
[[ ! -L "$DATA_ROOT" ]] || { echo "ERROR: refusing dataset/data symlink" >&2; exit 2; }
actual_data="$(realpath -m "$DATA_ROOT")"; expected_data="$(realpath -m "$EXPECTED_DATA_ROOT")"
[[ "$actual_data" == "$expected_data" && "$DATA_ROOT" == "$EXPECTED_DATA_ROOT" ]] || { echo "ERROR: data-root safety check failed" >&2; exit 2; }
[[ -f "$GENERATOR" ]] || { echo "ERROR: wrapper not found: $GENERATOR" >&2; exit 2; }
[[ -f "$SCRIPT_DIR/path_generator.py" ]] || { echo "ERROR: engine not found" >&2; exit 2; }
[[ -d "$MAPS_ROOT" ]] || { echo "ERROR: maps root not found: $MAPS_ROOT" >&2; exit 2; }
python3 -c 'import numpy, yaml, PIL, scipy' || { echo "ERROR: missing dependencies numpy, PyYAML, Pillow, SciPy" >&2; exit 2; }
read -r -a SCENARIOS <<< "$SCENARIOS_TEXT"
(("${#SCENARIOS[@]}">0)) || { echo "ERROR: scenario list is empty" >&2; exit 2; }
for scenario in "${SCENARIOS[@]}"; do
 [[ "$scenario" =~ ^E[0-9]{2}$ ]] || { echo "ERROR: invalid scenario $scenario" >&2; exit 2; }
 md="$(find "$MAPS_ROOT" -mindepth 1 -maxdepth 1 -type d -name "${scenario}_*" -print -quit)"
 [[ -n "$md" ]] || { echo "ERROR: no map directory for $scenario" >&2; exit 2; }
done
if ((PURGE_DATA)); then
 if ((ASSUME_YES==0)); then
  [[ -t 0 ]] || { echo "ERROR: non-interactive purge requires --yes after reviewing the deletion scope" >&2; exit 2; }
  printf 'Type PURGE to permanently delete all of %s: ' "$DATA_ROOT"; read -r confirmation
  [[ "$confirmation" == "PURGE" ]] || { echo "Cancelled; no files changed"; exit 1; }
 fi
 rm -rf -- "$DATA_ROOT"; echo "Removed: $DATA_ROOT"
elif ((ASSUME_YES)); then echo "ERROR: --yes is valid only with --purge-data" >&2; exit 2
fi
mkdir -p "$DATA_ROOT"
echo "DATASET GENERATION"; echo "Root: $DATA_ROOT"; echo "Scenarios: ${SCENARIOS[*]}"
if [[ "$MODE" == "goals" ]]; then echo "Targets: balanced total=$N_GOALS per scenario"; else echo "Targets: straight_semi_complex_complex_v_complex=$CLASS_QUOTA"; fi
echo "Timeout: ${TIMEOUT_SECONDS:-disabled}; attempts: ${MAX_ATTEMPTS:-unlimited}"
overall=0
for i in "${!SCENARIOS[@]}"; do
 scenario="${SCENARIOS[$i]}"
 scenario_number="${scenario#E}"
 seed=$((SEED_BASE + 10#$scenario_number))
 sd="$DATA_ROOT/$scenario/generation_seed_$seed"
 # Create only the scenario parent. The generator must create its own run_dir with
 # exist_ok=False; pre-creating sd makes every fresh run fail as an overwrite.
 mkdir -p "$DATA_ROOT/$scenario"
 log="$DATA_ROOT/$scenario/generation_seed_${seed}.log"
 cmd=(python3 "$GENERATOR" --scenario "$scenario" --seed "$seed" --run-name "generation_seed_$seed" --maps-root "$MAPS_ROOT" --output-dir "$DATA_ROOT" --progress-interval-seconds "$PROGRESS_INTERVAL_SECONDS")
 if [[ "$MODE" == "quota" ]]; then cmd+=(--st_sem_cmp_vc "$CLASS_QUOTA"); else cmd+=(--n-goals "$N_GOALS" --balance-class); fi
 [[ -z "$MAX_ATTEMPTS" ]] || cmd+=(--max-attempts "$MAX_ATTEMPTS")
 [[ -z "$TIMEOUT_SECONDS" ]] || cmd+=(--timeout-seconds "$TIMEOUT_SECONDS")
 echo "[$scenario] seed=$seed log=$log"
 set +e
 "${cmd[@]}" 2>&1 | tee "$log"
 status=("${PIPESTATUS[@]}")
 set -e
 if ((${status[0]:-1} != 0)); then
  echo "[$scenario] ERROR: generator exited with status ${status[0]}" >&2
  overall=1
  continue
 fi
 if [[ ! -s "$sd/path_pool.json" || ! -s "$sd/path_pool_poses.json" ]]; then echo "[$scenario] ERROR: missing pool/poses" >&2; overall=1; continue; fi
 validation=0
 python3 - "$sd/path_pool.json" "$sd/path_pool_poses.json" "$sd/report.json" <<'PY' || validation=$?
import collections,json,pathlib,sys
pp,posep,rp=map(pathlib.Path,sys.argv[1:])
try:
 pool=json.loads(pp.read_text()); poses=json.loads(posep.read_text()); report=json.loads(rp.read_text()) if rp.exists() else {}
except (OSError,json.JSONDecodeError) as exc: print(f"ERROR: invalid JSON {exc}",file=sys.stderr); raise SystemExit(2)
classes=("straight","semi_complex","complex","v_complex"); paths=pool.get("paths",[]); counts=collections.Counter(p.get("class") for p in paths); targets=pool.get("quota",{})
ids={str(p.get("id")) for p in paths}
if pool.get("pool_version")!=5 or not ids.issubset(poses): print("ERROR: malformed pool/poses",file=sys.stderr); raise SystemExit(2)
if not all(c in targets and int(targets[c])>0 for c in classes): print("ERROR: invalid target map",file=sys.stderr); raise SystemExit(2)
for c in classes: print(f"  {c}: {counts.get(c,0)}/{targets[c]}")
complete=all(counts.get(c,0)>=int(targets[c]) for c in classes)
print(f"  TOTAL: {len(paths)}/{sum(int(targets[c]) for c in classes)}; complete={complete}; stop_reason={report.get('stop_reason','unknown')}")
raise SystemExit(0 if complete else 1)
PY
 if ((validation==0)); then
  # The sampler contract is <data>/<scenario>/path_pool*.json.
  install -m 0644 "$sd/path_pool.json" "$DATA_ROOT/$scenario/path_pool.json"
  install -m 0644 "$sd/path_pool_poses.json" "$DATA_ROOT/$scenario/path_pool_poses.json"
  echo "[$scenario] COMPLETE; pool published to $DATA_ROOT/$scenario/"
 else
  echo "[$scenario] INCOMPLETE; partial pool retained only in $sd" >&2
  overall=1
 fi
done
python3 - "$DATA_ROOT" "$MODE" "$N_GOALS" "$CLASS_QUOTA" "$MAX_ATTEMPTS" "$TIMEOUT_SECONDS" "$PROGRESS_INTERVAL_SECONDS" "$SEED_BASE" "${SCENARIOS[@]}" <<'PY'
import collections,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); mode=sys.argv[2]; total=int(sys.argv[3]); quota_text=sys.argv[4]; ma=sys.argv[5]; timeout=sys.argv[6]; interval=float(sys.argv[7]); seed_base=int(sys.argv[8]); scenarios=sys.argv[9:]
classes=("straight","semi_complex","complex","v_complex")
manifest={"generator":"path_generator.py","pool_version":5,"target_mode":"balanced_total" if mode=="goals" else "explicit_class_quotas",
 "n_goals":total if mode=="goals" else None,"explicit_quotas":quota_text or None,
 "max_attempts_per_scenario":int(ma) if ma else None,"timeout_seconds_per_scenario":float(timeout) if timeout else None,
 "progress_interval_seconds":interval,"scenarios":{}}
for sc in scenarios:
 scenario_number=int(sc[1:])
 seed=seed_base+scenario_number
 run_dir=root/sc/f"generation_seed_{seed}"
 pp=run_dir/"path_pool.json"; rp=run_dir/"report.json"
 if not pp.is_file(): manifest["scenarios"][sc]={"status":"missing_pool"}; continue
 pool=json.loads(pp.read_text()); paths=pool.get("paths",[]); counts=collections.Counter(p.get("class") for p in paths); targets=pool.get("quota",{})
 report=json.loads(rp.read_text()) if rp.is_file() else {}
 complete=all(counts.get(c,0)>=int(targets.get(c,1)) for c in classes)
 manifest["scenarios"][sc]={"status":"complete" if complete else "incomplete","route_count":len(paths),
  "target_counts":{c:targets.get(c) for c in classes},"class_counts":{c:counts.get(c,0) for c in classes},
  "class_deficits":{c:max(0,int(targets.get(c,0))-counts.get(c,0)) for c in classes},
  "seed":pool.get("seed"),"run_directory":str(run_dir.relative_to(root)),"generation_complete":pool.get("generation_complete"),"stop_reason":report.get("stop_reason"),
  "max_attempts_hit":report.get("max_attempts_hit"),"timeout_hit":report.get("timeout_hit"),"elapsed_seconds":report.get("elapsed_seconds")}
manifest_path=root/"generation_manifest.json"; manifest_path.write_text(json.dumps(manifest,indent=2)); print(f"Manifest: {manifest_path}")
PY
if ((overall==0)); then echo "Dataset generation finished: all quotas complete"; else echo "Dataset generation finished with incomplete quotas; validated partial pools retained" >&2; fi
exit "$overall"
