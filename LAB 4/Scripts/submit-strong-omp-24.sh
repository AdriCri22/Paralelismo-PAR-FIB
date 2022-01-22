#!/bin/bash

#SBATCH --job-name=submit-strong-omp.sh
#SBATCH -D .
#SBATCH --output=submit-strong-omp.sh.o%j
#SBATCH --error=submit-strong-omp.sh.e%j

HOST=$(echo $HOSTNAME | cut -f 1 -d'.')

if (test "${HOST}" = "boada-1")
then
    echo "Use sbatch to execute this script"
    exit 0
fi

SEQ=multisort-seq
PROG=multisort-omp

size=32768
sort_size=1024
merge_size=1024
cutoff=16

np_NMIN=1
np_NMAX=24
N=3

# Make sure that all binaries exist
make $SEQ
make $PROG

HOST=$(echo $HOSTNAME | cut -f 1 -d'.')

out=$PROG-strong_${HOST}.txt	# File where you save the execution results
aux=./tmp.txt     	# archivo auxiliar

outputpath1=./speedup1.txt
outputpath2=./speedup2.txt
outputpath3=./elapsed.txt
rm -rf $outputpath1 2> /dev/null
rm -rf $outputpath2 2> /dev/null
rm -rf $outputpath3 2> /dev/null

echo -n Executing $SEQ sequentially > $out
elapsed=0  # Acumulacion del elapsed time de las N ejecuciones del programa
partial=0  # Acumulacion del multisort time de las N ejecuciones del programa

i=0        # Variable contador de repeticiones
while (test $i -lt $N)
do
    echo $'\n' >> $out
    /usr/bin/time --format=%e ./$SEQ -n $size -s $sort_size -m $merge_size >> $out 2>$aux

    time1=`cat $aux|tail -n 1`
    time2=`cat $out|tail -n 4  | grep "Multisort execution time"| cut -d':' -f 2`

    elapsed=`echo $elapsed + $time1|bc`
    partial=`echo $partial + $time2|bc`

    i=`expr $i + 1`
done

echo $'\n' >> $out
echo -n ELAPSED TIME AVERAGE OF $N EXECUTIONS = >> $out
sequential1=`echo $elapsed/$N|bc -l`
echo $sequential1 >> $out
echo -n MULTISORT TIME AVERAGE OF $N EXECUTIONS = >> $out
sequential2=`echo $partial/$N|bc -l`
echo $sequential2 >> $out

i=0
export OMP_WAIT_POLICY="passive"

PARS=$np_NMIN
while (test $PARS -le $np_NMAX)
do
    echo $'\n' >> $out
    echo -n Executing $PROG in parallel with $PARS threads >> $out
    elapsed=0  # Acumulacion del elapsed time de las N ejecuciones del programa
    partial=0  # Acumulacion del multisort time de las N ejecuciones del programa

    while (test $i -lt $N)
    do
        echo $'\n' >> $out
        export OMP_NUM_THREADS=$PARS
        /usr/bin/time --format=%e ./$PROG -n $size -s $sort_size -m $merge_size -c $cutoff >> $out 2>$aux

        time1=`cat $aux|tail -n 1`
        time2=`cat $out|tail -n 4  | grep "Multisort execution time"| cut -d':' -f 2`

        elapsed=`echo $elapsed + $time1|bc`
        partial=`echo $partial + $time2|bc`

        rm -f $aux
        i=`expr $i + 1`
    done

    echo $'\n' >> $out
    echo -n ELAPSED TIME AVERAGE OF $N EXECUTIONS = >> $out
    average1=`echo $elapsed/$N|bc -l`
    result1=`echo $sequential1/$average1|bc -l`
    echo $average1 >> $out
    echo -n MULTISORT TIME AVERAGE OF $N EXECUTIONS = >> $out
    average2=`echo $partial/$N|bc -l`
    result2=`echo $sequential2/$average2|bc -l`
    echo $average2 >> $out

    i=0

    #output PARS i average en fichero speedup1
    echo -n $PARS >> $outputpath1
    echo -n "   " >> $outputpath1
    echo $result1 >> $outputpath1

    #output PARS i average en fichero speedup2
    echo -n $PARS >> $outputpath2
    echo -n "   " >> $outputpath2
    echo $result2 >> $outputpath2

    #output PARS i average en fichero elapsed
    echo -n $PARS >> $outputpath3
    echo -n "   " >> $outputpath3
    echo $average2 >> $outputpath3

    #incrementa el parametre
    PARS=`expr $PARS + 1`
done

jgraph -P strong-omp.jgr >  $PROG-strong_${HOST}.ps
usuario=`whoami`
fecha=`date`
sed -i -e "s/UUU/$usuario/g" $PROG-strong_${HOST}.ps
sed -i -e "s/FFF/$fecha/g" $PROG-strong_${HOST}.ps

