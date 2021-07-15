echo "=================="

runtype=$1
country=$2
outdir=$3
model=$4

mkdir $outdir

if [ $runtype -eq 1 ] || [ $runtype -eq 2 ] || [ $runtype -eq 3 ] || [ $runtype -eq 4 ] || [ $runtype -eq 5 ]
then
    k=$5
    stream="../data/"$country"_cr.txt"
    
    if [ $model == 'LLD' ]
    then
        linearFit="True"
    elif [ $model == 'NLLD' ]
    then
        linearFit="False"
    else
        echo "ERROR: Specify the model: LLD or NLLD."
        exit 1
    fi
    
    if [ $runtype -eq 3 ]
    then
        errRat=$6
        python -W ignore main.py -s $stream -o $outdir -t $runtype -lf $linearFit -k $k -e $errRat
    else
        python -W ignore main.py -s $stream -o $outdir -t $runtype -lf $linearFit -k $k
    fi
elif [ $runtype -eq 6 ] || [ $runtype -eq 7 ] || [ $runtype -eq 8 ] || [ $runtype -eq 9 ] || [ $runtype -eq 10 ]
then
    if [ $model == 'SIR' ]
    then
        stream="../data/"$country"_sir.txt"
        if [ $runtype -eq 8 ]
        then
            errRat=$5
            python -W ignore main.py -s $stream -o $outdir -t $runtype -e $errRat
        else
            python -W ignore main.py -s $stream -o $outdir -t $runtype
        fi
    else
        echo "ERROR: The model does not match. runtype 6-10 should be SIR."
        exit 1
    fi
fi
