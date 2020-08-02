#!/bin/bash

#SBATCH --job-name=PhP
#SBARCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60GB#124GB
#SBATCH --time=9:00:00



#ppn=32,mem=512GB,walltime=12:00:00
#ppn=20,mem=189GB,walltime=12:00:00
#ppn=20,mem=189GB,walltime=8:00:00
#ppn=20,mem=62GB,walltime=2:00:00
#ppn=10,mem=8GB,walltime=10:00

module purge
#Py3 modules
module load anaconda3/5.3.1
module load scikit-learn/intel/0.18.1


cd /home/jc3832/PhenoPredict/PhP

srun python RicePredictor3xg_biomass.py -em XG

