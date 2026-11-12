#!/bin/bash
cd /home/duanyuhan/dyh/motion/lines/line14
/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python scripts_run/line14_sched.py scripts_run/line14_C9_jobs.txt --gpus 3,4,6,7,5 --log-dir logs/line14/C9 > logs/line14/C9_sched.log 2>&1
/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python scripts_run/line14_sched.py scripts_run/line14_C9e_jobs.txt --gpus 3,4,6 --log-dir logs/line14/C9e > logs/line14/C9e_sched.log 2>&1
