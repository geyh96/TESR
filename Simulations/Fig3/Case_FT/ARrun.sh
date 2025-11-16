#!/bin/bash

mkdir -p history
P_array=(30)
# NSource_array=(500 1000 1500 2000 2500 3000) 
NSource_array=(2000) 
NTarget_array=(300)
# NTarget_array=(200)
departure_array=(0 1)
S_array=(2 3 4 5 6 7 8)
njob=200   #simutaneously run these many jobs


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
            for ideparture in ${departure_array[*]}
            do
                for numS in ${S_array[*]}
                do
for iloop in $(seq 1 $Nall)
do

    # iloop=$(($vv + $idx_Begin - 1))
    ccount=$(($ccount+1))
    iflag=$(($ccount % $njob))
    echo -n "icount-"$ccount
    if [ $iflag -eq 0 ]
    then
    sleep 10s
        nohup  python3 -u $Target_file  --iloop  $iloop --numS $numS --P $P --NSource $NSource --NTarget $NTarget --ideparture $ideparture >> "history/Zresult"$iloop"_ideparture_"$ideparture"_numS_"$numS".log" 2>&1
    else         
        nohup  python3 -u $Target_file  --iloop  $iloop --numS $numS --P $P --NSource $NSource --NTarget $NTarget --ideparture $ideparture >> "history/Zresult"$iloop"_ideparture_"$ideparture"_numS_"$numS".log" 2>&1 &
    fi
done


                done
            done
        done
    done
done

wait
echo "All Done!"