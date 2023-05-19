#!/bin/bash

# ###### Zona de Parámetros de solicitud de recursos a SLURM ############################
#
#SBATCH --job-name=LQR_N1		#Nombre del job
#SBATCH -p short			#Cola a usar, Default=short (Ver colas y límites en /hpcfs/shared/README/partitions.txt)
#SBATCH -N 1				#Nodos requeridos, Default=1
#SBATCH -n 1				#Tasks paralelos, recomendado para MPI, Default=1
#SBATCH --cpus-per-task=8		#Cores requeridos por task, recomendado para multi-thread, Default=1
#SBATCH --mem=8000		#Memoria en Mb por CPU, Default=2048
#SBATCH --time=12:00:00			#Tiempo máximo de corrida, Default=2 horas
#SBATCH --mail-user=cd.contreras@uniandes.edu.co
#SBATCH --mail-type=ALL			
#SBATCH -o LQR_N1.o%j			#Nombre de archivo de salida
#
########################################################################################

# ################## Zona Carga de Módulos ############################################
module load anaconda/python3.9

pip3 install torch torchvision torchaudio

########################################################################################


# ###### Zona de Ejecución de código y comandos a ejecutar secuencialmente #############
host=`/bin/hostname`
date=`/bin/date`
echo "Soy un JOB de prueba"
echo "Corri en la maquina: "$host
echo "Corri el: "$date
python3 main.py
echo -e "Ejecutando Script de R \n"
echo -e "Finalicé la ejecución del script \n"
########################################################################################

