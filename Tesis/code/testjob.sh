#!/bin/bash

# ###### Zona de Parámetros de solicitud de recursos a SLURM ############################
#
#SBATCH --job-name=TestJOB		#Nombre del job
#SBATCH -p short			#Cola a usar, Default=short (Ver colas y límites en /hpcfs/shared/README/partitions.txt)
#SBATCH -N 1				#Nodos requeridos, Default=1
#SBATCH -n 1				#Tasks paralelos, recomendado para MPI, Default=1
#SBATCH --cpus-per-task=1		#Cores requeridos por task, recomendado para multi-thread, Default=1
#SBATCH --mem=2000		#Memoria en Mb por CPU, Default=2048
#SBATCH --time=00:10:00			#Tiempo máximo de corrida, Default=2 horas
#SBATCH --mail-user=USER@uniandes.edu.co
#SBATCH --mail-type=ALL			
#SBATCH -o TEST_job.o%j			#Nombre de archivo de salida
#
########################################################################################

# ################## Zona Carga de Módulos ############################################
module load  R/3.5.1mro

########################################################################################


# ###### Zona de Ejecución de código y comandos a ejecutar secuencialmente #############
sleep 60
host=`/bin/hostname`
date=`/bin/date`
echo "Soy un JOB de prueba"
echo "Corri en la maquina: "$host
echo "Corri el: "$date

echo -e "Ejecutando Script de R \n"
echo -e " Si tuviera un script lo ejecutaría asi: R CMD BATCH saludo.R \n"
echo -e "Finalicé la ejecución del script \n"
########################################################################################

