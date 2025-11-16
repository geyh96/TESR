#!/bin/bash


mkdir ./history

P_array=(60)
NSource_array=(2000) 
NTarget_array=(300)
dim_array=(8 16 32 64)
# NTarget_array=(200)

njob=100


idx_Begin=1
idx_End=100
Target_file="Case_Cor.py"

ncores=1



Nall=$(($idx_End - $idx_Begin + 1))
 # nbatch=$(($Nall / $njob + 1))
# nbatch_1=$(($nbatch - 1))
# njob_1=$(($njob - 1))


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
        for dim in ${dim_array[*]}
        do
for iloop in $(seq 1 $Nall)
do

    # iloop=$(($vv + $idx_Begin - 1))
    ccount=$(($ccount+1))
    iflag=$(($ccount % $njob))
    echo -n "icount-"$ccount
    if [ $iflag == 0 ]
    then
    sleep 10s
        nohup  python3 -u $Target_file  --iloop  $iloop --P $P --NSource $NSource --NTarget $NTarget --dim $dim >> "./history/Zresult"$iloop"_P_"$P"_NT_"$NTarget"_NS_"$NSource"_D_"$dim".log" 2>&1
    else         
        nohup  python3 -u $Target_file  --iloop  $iloop --P $P --NSource $NSource --NTarget $NTarget --dim $dim >> "./history/Zresult"$iloop"_P_"$P"_NT_"$NTarget"_NS_"$NSource"_D_"$dim".log" 2>&1 &
    fi
done



        done
    done
done
done
