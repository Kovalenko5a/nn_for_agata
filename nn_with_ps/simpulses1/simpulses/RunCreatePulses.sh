#!/bin/bash
namebase=/home/ljungvall/ljungvall/bases/AGATAGeFEMBasisCoreR6
for I in `find  ./Data/ -iname  '[0-9][RGB]' -or -iname '1[0-9][RGB]' | sort` 
do
    crystalname=`echo $I | sed -e 's#./Data/##'`
    N=${#crystalname}
    crystaltype=`echo $crystalname | cut -b $N-`
    crystal=`echo $crystalname | cut -b $(($N-1))`
    case $crystaltype in
	R*)
	    Fieldname=A001
	    Geomname=AGATARed.list
	    anisname=anisotropy_complete_a001
	    ;;
	G*)
	    Fieldname=B002
	    Geomname=AGATAGreen.list
	    anisname=anisotropy_complete_b002
	    ;;
	B*)
	    Fieldname=C002
	    Geomname=AGATABlue.list
	    anisname=anisotropy_complete_c002
	    ;;
    esac
    command="LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ljungvall/gammaSoftware/lib/  AGATAGeFEM_Writelibfield=yes AGATAGeFEM_NumberOfThreads=1 AGATAGeFEM_NoADF=yes mpirun -envall -np 5 AGATAGeFEM --AGATAsim -F${namebase}/${Fieldname}/${Fieldname} -A/home/ljungvall/Ljungvall/AGATAGeFEM/aniosotropyfiles/anisotropy_completeRotZ045deg -S${namebase}/${Fieldname}/${Geomname} -I${I}/AGATAGeFEMINPUT_${crystalname}_0000.lmevents -O${I}/Traces_${crystalname} 2>/dev/null"
    eval $command
done
