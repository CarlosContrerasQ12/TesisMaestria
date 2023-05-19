#!/bin/sh

#SBATCH -p gpu  	# Cola para correr el job.  Especificar gpu partition/queue (required)
#SBATCH --gres=gpu:1	# GPUs solicitadas (required), Default=1
#SBATCH -N 1		# Nodos requeridos, Default=1
#SBATCH -n 8		# Cores por nodo requeridos, Default=1
#SBATCH --mem=16G  	# Memoria Virtual/RAM, Default=2048
#SBATCH -t 01:00:00  	# Walltime 
#SBATCH --mail-user=user@uniandes.edu.co
#SBATCH --mail-type=ALL
#SBATCH --job-name=test-1GPU_8threads
#SBATCH -o job_1GPU8threads.log  # Output filename
host=`/bin/hostname`
date=`/bin/date`
echo "Soy un JOB de prueba en GPU"
echo "Corri en la maquina: "$host
echo "Corri el: "$date

echo "Voy a correr un programa que sirve para ver cual es la GPU que tengo instalada"
nvidia-smi

echo "Voy a correr un programa de prueba en gpu puedes ver el codigo fuente en: /hpcfs/shared/README/test.cu"
/hpcfs/shared/README/test
