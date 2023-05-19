Bienvenido a Hypatia, el servicio de HPC de la Universidad de los Andes.

Aquí te dejamos la información que debes saber antes de hacer uso del cluster

-------------------------------------------------
              SISTEMA OPERATIVO
-------------------------------------------------

Nuestro sistema está montado en Linux Centos 8 con OpenHPC

El manejador de trabajos es SLURM

El sistema de almacenamiento es Beegfs

El software está montado y disponible con Modules


------------------------------------------------
              EL ALMACENAMIENTO
------------------------------------------------

El almacenamiento compartido por usuarios corre sobre el sistema Beegfs, cuenta con 230Tb compartidas 
Los archivos de usuario se encuentran en /hpcfs/home/
Las carpetas compartidas se encuentran en /hpcfs/shared/
Este sistema no cuenta con Backup y solo deben mantenerse los archivos necesarios para el trabajo en el cluster

------------------------------------------------
              EL SOFTWARE
------------------------------------------------

El software disponible a través de modulos puede consultarse con la opción
	module avail
y puede ser cargado dentro del trabajo o sesión interactiva con
	module load <modulo>

------------------------------------------------
              EL MANEJADOR DE TRABAJOS
------------------------------------------------

El cluster funciona con el manejador de trabajos SLURM
Toda la documentación está disponible en https://slurm.schedmd.com

En la carpeta /hpcfs/shared/REAMDE se encuentran algunos ejemplos de scripts y documentación básica

Para jobs con cpus:  /hpcfs/shared/README/testcpu.sh
Para jobs con gpus:  /hpcfs/shared/README/testgpu.sh
Para jobs interactivos: /hpcfs/shared/README/srun.txt

Colas de trabajos: /hpcfs/shared/README/partitions.txt

Comandos básicos: /hpcfs/shared/README/slurmdoc.txt
