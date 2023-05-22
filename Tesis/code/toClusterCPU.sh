#!/bin/bash
ssh cd.contreras@hypatia.uniandes.edu.co 'rm main.py'
ssh cd.contreras@hypatia.uniandes.edu.co 'rm testjob.sh'
ssh cd.contreras@hypatia.uniandes.edu.co 'rm -r pdeSolver'
scp -r ./pdeSolver cd.contreras@hypatia.uniandes.edu.co:./pdeSolver
scp mainInterp.py cd.contreras@hypatia.uniandes.edu.co:./
scp testjob.sh cd.contreras@hypatia.uniandes.edu.co:./
ssh cd.contreras@hypatia.uniandes.edu.co 'sbatch testjob.sh'
