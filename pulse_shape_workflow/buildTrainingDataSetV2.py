#1. From read_only_gamma2.py we can receive the cut of data: 
# number of interactions, crystal, segment, bit of data
#2. Bit of data received from *.lmevents corespond to time in OnlyGammaEvents0000.txt
#See "take_time_of_bit.py" for more detaile
#3. If we know the bit of data, crystal, segment we can recive the PS
#4. After that we can create the snapshot matrix of detector in cell
# should be pulse shape - features case and number of interactions - lable.
#5. PS we can present in form of (Signal Ampl; Time) set of points or
# two vectors of A and T.


#

#First lets create the dictionary. For each time gate we should have separate detectors snapshot

from read_only_gammaV2 import mask_for_ps
from cut_ps_by_num import cut_ps_by_num
import numpy as np
from numpy import savez_compressed

def trainingDataSet(theDataSize=1000):
    def detectorName(detectorNum=0):
        num=str(detectorNum//3+1)
        BGR=["A","B","C"][detectorNum%3]
        detector = num+BGR
        return detector

    masks, globalTimeIndexes = mask_for_ps(theDataSize)
    print(masks)
    # But think in logick according which we start from 0 and end with N-1
    # 50 - for 50 points of signal amplitude; 1 - number of interactions (51 in total)
    # 3 - for RGB detectors of each stuck
    # 15 - number of stacks of detectors
    # 6 x 6 - sector * slice of each detector
    # agataSnapshot = np.zeros((51, 3, 16, 6, 6))
    
    k = 0
    gtIndex = 0
    # GLOBAL TIME START FROM ZERO!
    for gt in globalTimeIndexes:
        agataSnapshot = np.zeros((51, 3, 15, 6, 6))
        while masks[k][5]==gt and k<len(masks)-1:
            currentDetectorName = detectorName(masks[k][1])
            currentSegmentName = str(masks[k][2]) if  masks[k][2]>9 else "0"+str(masks[k][2])
            df = cut_ps_by_num(masks[k][0],currentDetectorName)
            amplitude = df[currentSegmentName].to_numpy()
            # Detector collor in bunch, bunch number, sector number, slice number
            position = [masks[k][1]%3, masks[k][1]//3, masks[k][2]%10, masks[k][2]//10]
            # First - add the number of interactions in to snapshot:
            agataSnapshot[50][position[0]][position[1]][position[2]][position[3]] = masks[k][3]
            # Second - insert the signal amplitude in to shapshot (We should take in account PS from each segment x sector of detector)
            for amplIndex in range(len(amplitude)):
                for ii in range(6):
                    for jj in range(6):
                        agataSnapshot[amplIndex][position[0]][position[1]][ii][jj] = df[str(ii)+str(jj)][amplIndex]
            k+=1
        savez_compressed('./compresed_dataset/gateTimeData'+str(gtIndex)+'.npz', agataSnapshot)
        # np.save('./dataset/gateTimeData'+str(gtIndex)+'.npy', agataSnapshot)
        gtIndex+=1
        print("done snapshot N ", gtIndex)
        
    return len(masks)
    
