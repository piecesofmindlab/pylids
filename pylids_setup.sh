printf "Make sure you have git setup and conda installed before proceeding \n"

read -p "Press [Enter] key to continue..."

# Environment setup
# Check for interactive environment
set -e

if [[ $- == *i* ]]
then
	echo 'Interactive mode detected.'
else
	echo 'Not in interactive mode! Please run with `bash -i pylids_setup.sh`'
	exit
fi

export name=pylids

# Assure mamba is installed
if hash mamba 2>/dev/null; then
		echo ">>> Found mamba installed."
        mamba update mamba
    else
        conda install mamba -n base -c conda-forge
		mamba update mamba
fi

# Create initial environment
ENVS=$(conda env list | awk '{print $name}' )
if [[ $ENVS = *"$name"* ]]; then
	echo "Found environment $name"
else
	mamba env create -f environment_pylids_py37.yml
fi;

# Activate it
conda activate $name

# Try to do this: (check OS first)
export unamestr=`uname`

# Make source code directory for git libraries
if [ -d ~/Code ]
	then
		echo "Code directory found."
	else
		echo "Creating ~/Code directory."
		mkdir ~/Code
fi
cd ~/Code/

# ... pylids (for dlc based pupil and eyelid detection)
echo "========================================"
echo ">>> Installing pylids"
if [ -d pylids ]
then
        :
else
        git clone git@github.com:piecesofmindlab/pylids.git
fi
cd pylids
python setup.py install
cd ..

# ... deeplabcut (pylids dependency)

echo "========================================"
echo ">>> Installing deeplabcut"
if [ -d deeplabcut ]
then
        :
else
        git clone git@github.com:arnabiswas/deeplabcut.git
fi
cd deeplabcut
git checkout pylids
python setup.py install
cd ..

cd ~

ipython kernel install --user --name $name
