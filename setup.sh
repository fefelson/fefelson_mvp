#!/bin/bash
sudo pacman -Syu
sudo pacman -S git base-devel python python-pip code nano sqlitebrowser firefox
python -m venv betting_env
source betting_env/bin/activate
pip install --upgrade pip
pip install numpy pandas scipy statsmodels scikit-learn matplotlib seaborn requests beautifulsoup4 lxml pulp cvxpy jupyter
echo "Setup complete! Activate with 'source betting_env/bin/activate'"