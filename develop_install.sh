#!/bin/bash

PROJECT_NAME="tfidf_from_scratch"
ZSH_BIN=$(which zsh)

# source conda and activate environment
if [ "$ZSH_BIN" != "" ]; then
  echo "zsh detected"
  conda init zsh && source "$HOME/.zshrc" && conda activate $PROJECT_NAME
else
  echo "ZSH is not installed, automatic installation failed. You can finish the installation manually by running: \n"
  echo "conda activate $PROJECT_NAME && conda install ipykernel -y && python3 -m ipykernel install --user --name=$PROJECT_NAME && make dev_install"
  echo "\n"
  exit 1
fi

python3 -m pip install -U pip-tools=="7.3.0" && pip install -U ipykernel
make dev_install
python3 -m ipykernel install --user --name=$PROJECT_NAME

