
_list_projects() {
  local EXP_DIR=${EXP_DIR:-"$HOME/experiments"}
  local -a options
  local context state state_descr line
  typeset -A opt_args

  _arguments \
    '1: :->level1' \
    '2: :->level2' \
    '3: :->level3'

  case $state in
    level1)
      options=($(ls "$EXP_DIR" 2>/dev/null))
      compadd "$@" -- $options
      ;;
    level2)
      options=($(ls "$EXP_DIR/${words[2]}" 2>/dev/null))
      compadd "$@" -- $options
      ;;
    level3)
      options=($(ls "$EXP_DIR/${words[3]}/${words[2]}" 2>/dev/null))
      compadd "$@" -- $options
      ;;
  esac
}

_train() {
  local -a yamls
  local basedir state

  # Resolve the directory of the script being completed
  # words[1] is the command typed (e.g., ./train.sh or path/to/train.sh)
  basedir=${words[1]:h}
  [[ -z $basedir || $basedir == $words[1] ]] && basedir=.
  [[ $basedir == . ]] && basedir=$PWD

  # Collect YAMLs (relative to the script dir), then present paths
  # relative to the current working directory for nicer display.
  yamls=(${(f)"$(command find "$basedir" -type f \( -name '*.yaml' -o -name '*.yml' \) 2>/dev/null)"})
  # De-duplicate and map to paths relative to $PWD
  typeset -aU yamls
  yamls=("${yamls[@]/#${PWD}\//}")

  _arguments -C \
    '1:config file:->cfg' \
    '2:mode:(local pm2 slurm)'

  case $state in
    cfg)
      # Offer the found YAMLs as first-arg candidates
      compadd -Q -S ' ' -- "${yamls[@]}"
      return 0
      ;;
  esac

  return 0
}

# attach
compdef _train train.sh
compdef _list_projects delete_experiment_safely.sh
compdef _list_projects copy_experiment.sh

export TRADING_AUTO_COMPLETE=1
