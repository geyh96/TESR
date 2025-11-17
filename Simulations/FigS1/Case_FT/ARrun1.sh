#!/bin/bash

mkdir -p history
P_array=(10 20 30)
NSource_array=(500 1000 1500 2000 2500) 
# NSource_array=(1000) 
# NTarget_array=(100 150 200 250 300)
NTarget_array=(500)

#3*6*1
#18

njob=200   #simutaneously run these many jobs
Target_file="Case_Cor.py"

ncores=1
Nall=100





export OMP_NUM_THREADS=$ncores
export OPENBLAS_NUM_THREADS=$ncores
export MKL_NUM_THREADS=$ncores
export NUMEXPR_NUM_THREADS=$ncores
export VECLIB_MAXIMUM_THREADS=$ncores

ccount=0
for P in ${P_array[*]}
do

    for NSource in ${NSource_array[*]}
    do
        for NTarget in ${NTarget_array[*]}
        do

for iloop in $(seq 1 $Nall)
do


    ccount=$(($ccount+1))
    iflag=$(($ccount % $njob))
    echo -n "icount-"$ccount
    if [ $iflag == 0 ]
    then
    sleep 10s
        nohup  python3 -u $Target_file  --iloop  $iloop --P $P  --NSource $NSource --NTarget $NTarget >> "history/ZZresult"$iloop"_P_"$P"_NT_"$NTarget"_NS_"$NSource".log" 2>&1
    else         
        nohup  python3 -u $Target_file  --iloop  $iloop --P $P   --NSource $NSource --NTarget $NTarget >> "history/ZZresult"$iloop"_P_"$P"_NT_"$NTarget"_NS_"$NSource".log" 2>&1 &
    fi
done



        done
    done

done

wait 
echo "All Done!"