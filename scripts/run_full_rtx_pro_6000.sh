#!/bin/bash
# Run the five-seed, full-data experiment on one CURC NVIDIA RTX PRO 6000.
# Submit this file from the repository root after creating the logs directory.

#SBATCH --job-name=recsys-full-gpu
#SBATCH --account=ucb-general
#SBATCH --partition=artxpro6000
#SBATCH --qos=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --output=logs/recsys-full-gpu-%j.out
#SBATCH --error=logs/recsys-full-gpu-%j.err

set -euo pipefail

PERSISTENT_PROJECT="${SLURM_SUBMIT_DIR}"
RUN_DIR="/scratch/alpine/${USER}/rec-sys-full-gpu-${SLURM_JOB_ID}"
GPU_ENV="${REC_SYS_GPU_ENV:-/projects/${USER}/software/uv/envs/rec-sys-gpu}"
RESULT_SUBDIR="full_rtx_pro_6000"
ARTIFACT_DIR="${RUN_DIR}/artifacts/${RESULT_SUBDIR}"

if [[ ! -f "${PERSISTENT_PROJECT}/src/recommender_optimization_experiments.ipynb" ]]; then
    echo "Submit this script from the rec-sys-nn-optimization repository root." >&2
    exit 1
fi
if [[ ! -x "${GPU_ENV}/bin/python" ]]; then
    echo "GPU environment Python not found: ${GPU_ENV}/bin/python" >&2
    exit 1
fi

mkdir -p "${RUN_DIR}" "${ARTIFACT_DIR}"

# Preserve any partial diagnostics if setup or notebook execution fails.
copy_outputs_back() {
    status=$?
    trap - EXIT
    set +e
    printf '%s\n' "${status}" > "${ARTIFACT_DIR}/exit-status.txt"
    mkdir -p \
        "${PERSISTENT_PROJECT}/results/${RESULT_SUBDIR}" \
        "${PERSISTENT_PROJECT}/artifacts/${RESULT_SUBDIR}"
    if [[ -d "${RUN_DIR}/results/${RESULT_SUBDIR}" ]]; then
        rsync -a \
            "${RUN_DIR}/results/${RESULT_SUBDIR}/" \
            "${PERSISTENT_PROJECT}/results/${RESULT_SUBDIR}/"
    fi
    rsync -a \
        "${ARTIFACT_DIR}/" \
        "${PERSISTENT_PROJECT}/artifacts/${RESULT_SUBDIR}/"
    echo "Job exit status: ${status}"
    echo "Finished: $(date --iso-8601=seconds)"
    exit "${status}"
}
trap copy_outputs_back EXIT

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Persistent project: ${PERSISTENT_PROJECT}"
echo "Scratch run directory: ${RUN_DIR}"

# Stage a clean working copy. Existing quick-mode results remain in /projects
# and the full run writes to its own result subdirectory.
rsync -a \
    --exclude 'artifacts/' \
    --exclude 'results/full_rtx_pro_6000/' \
    "${PERSISTENT_PROJECT}/" \
    "${RUN_DIR}/"

module purge
module load uv
source "${GPU_ENV}/bin/activate"

cd "${RUN_DIR}"
mkdir -p "results/${RESULT_SUBDIR}" "${ARTIFACT_DIR}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export REC_SYS_QUICK_MODE=false
export REC_SYS_REQUIRE_CUDA=true
export REC_SYS_EXPECTED_CUDA_DEVICE="RTX PRO 6000"
export REC_SYS_REQUIRED_CUDA_ARCH="sm_120"
export REC_SYS_RUN_LABEL="full_rtx_pro_6000"
export REC_SYS_RESULTS_FILE="${RESULT_SUBDIR}/experiment_results.csv"

which python
python --version
uv pip check
python -m unittest discover -s tests -v

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; assert 'RTX PRO 6000' in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0); assert 'sm_120' in torch.cuda.get_arch_list(), torch.cuda.get_arch_list(); print('GPU:', torch.cuda.get_device_name(0)); print('Capability:', torch.cuda.get_device_capability(0)); print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('Architectures:', torch.cuda.get_arch_list()); x=torch.randn(2048, 2048, device='cuda'); y=x @ x; torch.cuda.synchronize(); print('GPU matrix test passed:', y.device)" \
    | tee "${ARTIFACT_DIR}/pytorch-device.txt"

uv pip freeze > "${ARTIFACT_DIR}/requirements-freeze.txt"
nvidia-smi -q > "${ARTIFACT_DIR}/nvidia-smi.txt"
lscpu > "${ARTIFACT_DIR}/hardware.txt"
scontrol show job "${SLURM_JOB_ID}" > "${ARTIFACT_DIR}/slurm-job.txt"
git rev-parse HEAD > "${ARTIFACT_DIR}/git-commit.txt" 2>/dev/null || true

echo "Starting five-seed full-data notebook: $(date --iso-8601=seconds)"
python -m nbconvert \
    --to notebook \
    --execute src/recommender_optimization_experiments.ipynb \
    --output full_rtx_pro_6000_executed.ipynb \
    --output-dir "${ARTIFACT_DIR}" \
    --ExecutePreprocessor.timeout=-1

python -c "import pandas as pd; from pathlib import Path; path=Path('results/full_rtx_pro_6000/experiment_results.csv'); rows=pd.read_csv(path); expected={42,123,456,789,2026}; actual=set(rows['seed'].astype(int)); assert actual == expected, (actual, expected); assert set(rows['quick_mode'].astype(str).str.lower()) == {'false'}; assert set(rows['compute_device']) == {'cuda'}; assert set(rows['run_complete'].astype(str).str.lower()) == {'true'}; print('Validated rows:', len(rows)); print('Seeds:', sorted(actual)); print('Result:', path)" \
    | tee "${ARTIFACT_DIR}/result-validation.txt"

echo "Full GPU experiment completed successfully."
