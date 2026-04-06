#!/bin/bash

red=$(tput setaf 1)
blue=$(tput setaf 4)
reset=$(tput sgr0)

main() {
  if [ -z "$TRADING_AUTO_COMPLETE" ]; then
    echo "${red} auto_complete.zsh not sourced. Run:${reset}"
    echo "${red} source ./auto_complete.bash${reset}"
    exit 1
  fi
  if [ -z "$BUNDLE_DIR" ]; then
    echo "${red} BUNDLE_DIR not defined"
    exit 1
  fi

  # check config file
  config_path=$1
  if [ -z "$config_path" ]; then
    echo "${red} First argument must be config (yaml) file.${reset}"
    exit 1
  fi

  if [ ! -f "$config_path" ]; then
    echo "${red} Config file does not exist: ${config_path} ${reset}"
    exit 1
  fi

  # check execution mode (local or remote)
  if [ -z "$2" ]; then
    echo "${red} Second argument must be execution mode (local/remote).${reset}"
    exit 1
  fi

  # parse yaml
  parse_yaml $config_path
  echo $experiment_name
  if [ -z "$experiment_name" ]; then
    echo "${red} Cannot parse experiment name from config.${reset}"
    exit 1
  fi
  echo -e "\n********************"
  echo "${blue}NAME: ${experiment_name}${reset}"
  echo -e "********************\n"

  # make git snapshot
  snapshot_git_commit $experiment_name

  # save snapshot
  backup_git_snapshot

  # binary executable
  exe=$(realpath "./train.py")

  if [ $2 == "local" ]; then
    python $exe --config_path $config_path
  elif [ $2 == "pm2" ]; then
    pm2 start "$exe" \
      --name "$experiment_name" \
      --no-autorestart \
      --interpreter ../.venv/bin/python \
      -- --config_path "$config_path"
    sleep 1s
    pm2 log "$experiment_name"
  elif [ $2 == "slurm" ]; then
    sbatch --export=ALL,EXP_NAME=${experiment_name} ./slurm.sh
  fi

}

function parse_yaml {
  experiment_name=$(cat_yaml.py --config_path $1)
}

# version control, save snapshot
snapshot_git_commit() {
  # check if in git tree
  inside_git_repo="$(git rev-parse --is-inside-work-tree 2>/dev/null)"
  if [ ! "$inside_git_repo" ]; then
    echo "${red} Must be inside a git tree."
    exit 1
  fi
  name=$1
  git add -A
  git add -f "${config_path}"
  branch_exists_already=$(git rev-parse --verify --quiet ${name})
  if [ -n "${branch_exists_already}" ]; then # exists already
    # Prompt replacement
    eval git diff "$name" --stat
    read -p "Replace branch? (y/n): " confirm && [[ $confirm == [yY] ]]  || return # return if denied

    branch_tmp="${name}-overwrite"
    git switch -c "$branch_tmp"
    git commit -m "$name"
    git branch -D "$name"
    git branch -m "$branch_tmp" "$name"
    echo -e "Branch replaced, branch=${name}"
  else
    # create new branch
    git checkout -b "$name"
    git commit -m "$name" # save snapshot/commit to branch
  fi
  # cleanup
  git push origin "${name}" -f &

}

backup_git_snapshot() {
  echo "[GitSnapshot] Backing up..."
  git bundle create /tmp/torch_examples.bundle --all
  cp /tmp/torch_examples.bundle $BUNDLE_DIR
  echo "[GitSnapshot] Done."
}

main "$@"

