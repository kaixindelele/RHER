#!/bin/bash

# Script to reproduce results

envs=(
	"FetchDrawer-v1"
	)

for ((i=0;i<5;i+=1))
do 
	for env in ${envs[*]}
	do
		mpirun -np 19 python -m baselines.run_bash_her \
		--env $env \
		--seed $i \
		--gpu_id=-1
	done
done
