#!/usr/bin/python2.7
import os
import os.path
import shutil
import sys
import optparse
simoutputfile="GammaEvents.0000" # to be changed if needed
dirstomake=["Global"]
positioncrystals=[""]
dirnames=["R","G","B","B"]
with open(simoutputfile,"r") as simfile:
    for aline in simfile:
        if "$" in aline: 
            break
        #Find crystal_lut
        if "CRYSTAL_LUT" in aline:
            for asecondline in simfile:
                if "ENDCRYSTAL_LUT" in asecondline:
                    break
                clustcrist=[int(word) for word in asecondline.split()]
                clust=clustcrist[0]/3+1
                if clustcrist[1]<0:
                    continue
                crist=dirnames[clustcrist[1]]
                dirstomake.append(str(clust)+crist)
        #Look for ancillary
        if "ANCIL" in aline:
            dirstomake.append("Ancillary")
        #Find CrystalPositionLookUpTable
        if "POSITION_CRYSTALS" in aline:
            for asecondline in simfile:
                if "ENDPOSITION_CRYSTALS" in asecondline:
                    break
                positioncrystals.append(asecondline)
basedir=["Conf","Data","Out"]
for doing in basedir:
    if not os.path.isdir(doing):
        os.makedirs(doing)
    for atdir in dirstomake:
        realname = doing+"/"+atdir
        if not os.path.isdir(realname):
            os.makedirs(realname)
            print "Makeing "+realname
#Copying CrystalLookUpTable
with open("Conf/Global/CrystalPositionLookUpTable","w") as cltfile:
    for oneline in positioncrystals:
        cltfile.write(oneline)
#Make conf files for BasicAFP
for atdir in dirstomake:
    if atdir!="Global" and atdir!="Ancillary":
        totalfilename = "Conf/"+atdir+"/BasicAFP.conf"
        with open(totalfilename,"w") as conffile:
            conffile.write("/data2/joa/SourceAtGSI/Data/"+atdir+"\n")
            conffile.write(" PSA_"+atdir+"_ 0\n")
