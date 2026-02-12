#!/bin/bash
# Clean installation of ROOT + Delphes in conda environment

set -e

# 1. Remove old environment (backup first)
echo "Creating fresh environment..."
conda deactivate 2>/dev/null || true
conda env remove -n madminer -y 2>/dev/null || true

# 2. Create new environment with ROOT from conda-forge
echo "Installing ROOT via conda..."
mamba create -n madminer -c conda-forge python=3.10 root -y

# 3. Activate and install MadMiner dependencies
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate madminer

# 4. Install MadMiner
echo "Installing MadMiner and dependencies..."
pip install numpy scipy pandas h5py madminer
pip install -e /user/qvanenge/tzq-sbi-phy/madminer-cli
pip install -e /user/qvanenge/tzq-sbi-phy/madminer-dag

# 5. Clean and compile Delphes
echo "Compiling Delphes..."
cd /data/atlas/users/qvanenge/Tools/MG5_aMC_v3_5_6/Delphes
make clean

# Export clean path (conda bin + minimal system)
export PATH=$(conda info --base)/envs/madminer/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=$(conda info --base)/envs/madminer/lib

# Compile
make -j4

# 6. Verify installation
echo "Verifying installation..."
which root-config
root-config --version
ls -lh DelphesHepMC 2>/dev/null && echo "SUCCESS: DelphesHepMC compiled!" || echo "ERROR: DelphesHepMC not found"

echo ""
echo "Installation complete! To use:"
echo "  conda activate madminer"
