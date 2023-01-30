#! /bin/sh
echo ./simpulses/Data/10B/Traces_10B.events

for i in {1..16..1}
do
	for j in A B C
do
	if [ $i -gt 9 ]
	then
		./eventstoascii ../Data/$i$j/Traces_$i$j.events
	fi
	if [ $i -lt 10 ]
	then
		./eventstoascii ../Data/0$i$j/Traces_0$i$j.events
	fi
	cp ./test1.csv ./csvPsDataBase/csvOut$i$j.csv
done
done