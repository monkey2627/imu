#!/bin/bash
cd /home/duanyuhan/dyh/motion/lines/line14
/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python scripts_run/line14_sched.py scripts_run/line14_C7a_jobs.txt --gpus 0,1,2,3,4,6 --log-dir logs/line14/C7a > logs/line14/C7a_sched.log 2>&1
/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python scripts_run/line14_sched.py scripts_run/line14_C7b_jobs.txt --gpus 0,1,2,3,4,6 --log-dir logs/line14/C7b > logs/line14/C7b_sched.log 2>&1
