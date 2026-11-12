#!/bin/bash
cd /home/duanyuhan/dyh/motion/lines/line14
/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python scripts_run/line14_sched.py scripts_run/line14_C9r_jobs.txt --gpus 0,1,2 --log-dir logs/line14/C9r > logs/line14/C9r_sched.log 2>&1
/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python scripts_run/line14_sched.py scripts_run/line14_C9re_jobs.txt --gpus 0,1,2 --log-dir logs/line14/C9re > logs/line14/C9re_sched.log 2>&1
