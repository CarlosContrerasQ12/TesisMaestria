#!/bin/bash
scp -r ./pdeSolver cd.contreras@hypatia.uniandes.edu.co:./pdeSolver
scp main.py cd.contreras@hypatia.uniandes.edu.co:./
scp testjob.sh cd.contreras@hypatia.uniandes.edu.co:./
ssh cd.contreras@hypatia.uniandes.edu.co 'sbatch testjob.sh'
