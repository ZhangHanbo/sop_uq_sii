# Create, repair and activate this package's conda env (name and packages: environment.yml).
# Source it from bash or zsh, in any folder:
#   source sop_uq_candidate/conda_env.sh
# A missing env is created (a few minutes, once). An env that cannot import this checkout's
# package is updated from environment.yml again. Then the env is activated.

if [ -n "${BASH_VERSION:-}" ] && [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "Source this file so the env stays active: source $0" >&2
  exit 1
fi

_sop_conda_env() {
  local here="$1" name sh=bash
  local check='import pathlib, sys, baselines, ipykernel, notebook, test_client
assert pathlib.Path(baselines.__file__).parents[1].samefile(sys.argv[1]), baselines.__file__'
  name="$(sed -n 's/^name: *//p' "$here/environment.yml")"
  command -v conda >/dev/null || { echo "conda not found: install Miniconda or Miniforge first" >&2; return 1; }
  [ -n "${ZSH_VERSION:-}" ] && sh=zsh
  # 'conda activate' needs conda's shell function; load it where 'conda init' has not run.
  typeset -f conda >/dev/null || eval "$(conda "shell.$sh" hook)"

  # --stack puts the env's programs first on PATH. Plain activation reuses the slot of the
  # active env, which can sit behind ~/.local/bin and so run another Jupyter.
  [ "${CONDA_DEFAULT_ENV:-}" = "$name" ] && conda deactivate
  if ! conda activate --stack "$name" 2>/dev/null; then
    echo "Creating conda env $name from $here/environment.yml"
    conda env create -f "$here/environment.yml"
    conda activate --stack "$name" || return 1
  fi
  if ! python -I -c "$check" "$here/src" 2>/dev/null; then
    echo "Conda env $name cannot import this package; updating it from environment.yml"
    conda env update -n "$name" -f "$here/environment.yml" && python -I -c "$check" "$here/src" || {
      echo "Still failing: remove it with 'conda env remove -n $name' and source this file again" >&2
      return 1
    }
  fi
  echo "Conda env $name is active ($(python --version))"
}

_sop_conda_env "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" >/dev/null && pwd)"
