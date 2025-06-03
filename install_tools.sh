#!/bin/bash
# filepath: protein-prediction/install_tools.sh
# chmod +x install_tools.sh

set -e

# Download and install Miniconda if conda is not available
if ! command -v conda &> /dev/null; then
    echo "Miniconda not found. Installing Miniconda..."
    wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash miniconda.sh -b -p "$HOME/miniconda"
    export PATH="$HOME/miniconda/bin:$PATH"
    source "$HOME/miniconda/etc/profile.d/conda.sh"
else
    echo "Conda found."
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# Remove old env if it exists (optional, for stateless sessions)
conda env remove -n protein-prediction -y || true

# Create and activate environment
conda create -n protein-prediction python=3.10 -y
conda activate protein-prediction

# Install tools and libraries
conda install -c bioconda emboss -y
conda install -c conda-forge -c bioconda mmseqs2 -y
pip install graph-part pandas scikit-learn biopython

echo "All tools and libraries installed in the 'protein-prediction' environment."
echo "To use it, run: source \$HOME/miniconda/etc/profile.d/conda.sh && conda activate protein-prediction"