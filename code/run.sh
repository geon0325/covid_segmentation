echo "=================="
echo "=================="

countryList="panama"
linearFit="False"

IFS=,
for country in $countryList;
do
	stream="../data/"$country"_cr.txt"
	outdir="./"
	mkdir $outdir
	echo $stream
	echo "-------------------"
	python -W ignore main.py -s $stream -o $outdir -t 2 -lf $linearFit;
done



echo "=================="
echo "        END       "
echo "=================="




