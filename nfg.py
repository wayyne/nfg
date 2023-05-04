#!/usr/bin/env python
#
#               (   (( . : (    .)   ) :  )     *Dept. of Geology &
#                (   ( :  .  :    :  )  ))          **Dept. of Chemistry,
#                 ( ( ( (  .  :  . . ) )        University of South Florida 
#                  ( ( : :  :  )   )  )
#                   ( :(   .   .  ) .'
#                    '. :(   :    )              /$$   /$$ /$$$$$$$$ /$$$$$$   
#                      (   :  . )  )           | $$$ | $$| $$_____//$$__  $$   
#                       ')   :   #@##          | $$$$| $$| $$     | $$  \__/   
#                      #',### \" #@  #@        | $$ $$ $$| $$$$$  | $$ /$$$$   
#                     #/ @'#~@#~/\   #         | $$  $$$$| $$__/  | $$|_  $$   
#                   ##  @@# @##@  `..@,        | $$\  $$$| $$     | $$  \ $$   
#                 @#/  #@#   _##     `\        | $$ \  $$| $$     |  $$$$$$/   
#               @##;  `#~._.' ##@      \_      |__/  \__/|__/      \______/    
#             .-#/           @#@#@--,_,--\                                     
#            / `@#@..,     .~###'         `~.            Version 38
#          _/         `-.-' #@####@          \                                 
#       __/     &^^       ^#^##~##&&&   %     \_     *Nikola Rogic, Ph.D.      
#      /       && ^^      @#^##@#%%#@&&&&  ^    \     *Franco Villegas-Garin 
#    ~/         &&&    ^^^   ^^   &&&  %%% ^^^   `~_   **Guy Dayhoff II
# .-'   ^^    %%%. &&   ___^     &&   && &&   ^^    \             
#/akg ^^^ ___&&& X & && |n|   ^ ___ %____&& . ^^^^^ `~. 
#        |M|       ---- .  ___.|n| /\___\                $Dec 2021 - May 2022$
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from os import listdir, system, remove, rename
from os.path import isfile, join
import subprocess
import sys
import time
import math
import numpy as np
import multiprocessing as mp
from os.path import exists
from osgeo import gdal, gdalconst, osr


#------------------------------------------------------------------------------
#  NFG run parameters
#
#  INPUT_PATH: .tif containing backscatter amplitude data, or a directory***
#  ANGLE_FILE: .txt containing lat,lon,slant angle,incidence angle
#  PINCH_INPUT: clip the input data using the pinch factor
#  AVERAGE_INPUTS: given a directory of input, yield a single output
#  CONVERT_INPUT_TO_BS: toggle to convert amplitudes to backscatter
#  USE_OG_BS_EQN: use the first equation provided by NASA to convert to BS
#  IMAGE_NOISE_BIAS: the noise in the raw data to be used for BS conversion
#  SPLIT_OUTPUTS: produce two outputs contain values opposite of the threshold
#  SPLIT_THRESHOLD: the value to use for spliting output into two
#
#  COHERENT_OUTPUT: treat directory of inputs as a single input for rescaling*
#  RESCALE_OUTPUT: rescale final output to the range [0,1]
#  CLUSTER_OUTPUT: based on the class boundary perform binary classificaion
#  CLASS_BOUNDARY: the boundary between smooth and rough used for clustering
#
#  DUMP_DATA: create a .dat column file containing all output pixel values
#  DUMP_NODATA: include NODATA_VALUE pixels when dumping data
#
#  ROUGHNESS_MODEL: the index of the roughness model for the run
#  WINDOWS: the dimensions for the NxN window** used for computing roughness
#  REPORT_AS_LOG10: toggle for reporting roughness as log10(roughness)
#  POPULATION_STATISTICS: toggle between N & N-1 denominators for stddev etc.
#
#  INTENSITY_WEIGHTED: weight final output using inverted backscatter values
#  PINCH_WEIGHTS: apply the pinch factor to constrain intensity weights
#  PINCH_FACTOR: the number of standard deviations to constrain data to
#
#  NODATA_VALUE: the NODATA_VALUE used in the input and the output
#  NODATA_ANGLE: the magic number used in the angle file indicating nodata
#  AVOID_BLANKS: do not compute roughness anytime the window is missing data
#  WORKAROUND_BLANKS: recursively reduce window size to remove missing data
#
#  MULTIPROCESSING: toggle for enabling multiprocessing
#
#  *also applies to intensity weighting
#
#  **multiple windows can be specified and the final output will be the avg
#    over all windows for a given pixel
#
#  ***if a directory is specifed, the output will be an average computed over
#     each tif contained in the directory if AVERAGE_INPUTS is True, otherwise
#     each tif contained therein will be treated independently
#
#------------------------------------------------------------------------------

#i/o
INPUT_PATH              = "smol.tif"
ANGLE_FILE              = None
PINCH_INPUT             = False
AVERAGE_INPUTS          = False
CONVERT_INPUT_TO_BS     = True
USE_OG_BS_EQN           = False
IMAGE_NOISE_BIAS        = 0.0
SPLIT_OUTPUT            = False
SPLIT_THRESHOLD         = .5 

COHERENT_OUTPUT         = False
RESCALE_OUTPUT          = False
CLUSTER_OUTPUT          = False
CLASS_BOUNDARY          = 0

DUMP_DATA               = False
DUMP_NODATA             = False

#roughness_model
ROUGHNESS_MODEL         = 7
WINDOWS                 = [3]
APPLY_GAUSSIAN_BLUR     = False
BLUR_FACTOR             = 1/np.sqrt(2*np.pi) #yields a value of 1 for pixel p_i
REPORT_AS_LOG10         = False
POPULATION_STATISTICS   = True

#intensity_weighting
INTENSITY_WEIGHTED      = False
PINCH_WEIGHTS           = True
PINCH_FACTOR            = 2.0

#missing_data
NODATA_VALUE            = -34028
NODATA_ANGLE            = -9999.000000
AVOID_BLANKS            = True
WORKAROUND_BLANKS       = False

#performance
MULTIPROCESSING         = True       


#------------------------------------------------------------------------------
# NFG models
# 
# @ Models always take a single input, W, which is a NxN window positioned 
#   such that the pixel, p_i, is at its center
# 
# @ Given a window, W_i, models return a measurement for pixel p_i
# 
# @ Models MUST be defined as m#, where # is the model's index e.g. m0, m1,..
# 
# @ Models are invoked by specifying a valid model index in the run parameters
#
#------------------------------------------------------------------------------

#model 0: m0_i = stddev(W_i) [population stddev OR sample stddev]
def m0(W):
    dW = W-np.mean(W)
    if POPULATION_STATISTICS is True:
        return np.sqrt(np.sum(dW**2)/(np.shape(W)[0]**2))
    else:
        return np.sqrt(np.sum(dW**2)/(np.shape(W)[0]**2-1))


#model 1: m1_i = mean_error(W_i - p_i)
def m1(W):
    index = int(np.floor(np.shape(W)[0]/2))
    dW = W - W[index,index]
    return np.sum(dW)/(np.shape(W)[0]**2)


#model 2: m2_i = MSE(W_i) w.r.t. p_i
def m2(W):
    index = int(np.floor(np.shape(W)[0]/2))
    dW = (W - W[index,index])**2
    return np.sum(dW)/(np.shape(W)[0]**2)


#model 3: m3_i = RMSE(W_i) w.r.t. p_i
def m3(W):
    index = int(np.floor(np.shape(W)[0]/2))
    dW = (W - W[index,index])**2
    return np.sqrt(np.sum(dW)/(np.shape(W)[0]**2))


#model 4: m4_i = stderr(W_i) [note: employs m0]
def m4(W):
    return m0(W)/np.sqrt(np.shape(W)[0]**2)


#model 5: m5_i = var(W_i) [population variance OR ubiased sample variance]
def m5(W):
    dW = W-np.mean(W)
    if POPULATION_STATISTICS is True: 
        return (np.sum(dW**2))/(np.shape(W)[0]**2)
    else: 
        return (np.sum(dW**2))/(np.shape(W)[0]**2-1)
    

#model 6: m6_i = HI(W_i) [hypsometric integral] 
def m6(W):
    amin=np.amin(W)
    return (np.mean(W)-amin)/(np.amax(W)-amin)


#model 7: m7_i = mean_absolute_deviation(W_i) 
def m7(W):
    dW = (W-np.mean(W))*filter
    return np.sum(np.abs(dW))/(np.shape(W)[0]**2) 


#model 8: m8_i = mean(W)
def m8(W):
    return np.mean(W)


#model 9: m9_i = fill_in_de_blanks
def m9(W):
    index = int(np.floor(np.shape(W)[0]/2))
    if W[index,index] != NODATA_VALUE:
        return W[index,index]
    return np.mean(W[np.where(W!=NODATA_VALUE)])


#------------------------------------------------------------------------------
# filter kernals
#------------------------------------------------------------------------------

filter=np.ones(shape=(WINDOWS[0],WINDOWS[0]))

#kernal 1: 2D guassian filter kernel
def k0(x,y,sigma):
    return (1/(2*np.pi*(sigma**2)))*np.exp(-((x**2+y**2)/(2*(sigma**2))))


#------------------------------------------------------------------------------
# ancilliary functions 
#------------------------------------------------------------------------------

#convert DD coordinates to DMS coordinates
def dd2dms(longitude, latitude):
    
    split_degx = math.modf(longitude)
    degrees_x = int(split_degx[1])
    minutes_x = abs(int(math.modf(split_degx[0] * 60)[1]))
    seconds_x = abs(round(math.modf(split_degx[0] * 60)[0] * 60,2))
    
    split_degy = math.modf(latitude)
    degrees_y = int(split_degy[1])
    minutes_y = abs(int(math.modf(split_degy[0] * 60)[1]))
    seconds_y = abs(round(math.modf(split_degy[0] * 60)[0] * 60,2))
    
    # account for E/W & N/S
    if degrees_x < 0:
        EorW = "W"
    else:
        EorW = "E"
    
    if degrees_y < 0:
        NorS = "S"
    else:
        NorS = "N"
    
    return "\t" + str(abs(degrees_x)) + u"\u00b0 " + str(minutes_x) + \
            "' " + str(seconds_x) + "\" " + EorW+", " + str(abs(degrees_y)) + \
            u"\u00b0 " + str(minutes_y) + "' " + str(seconds_y) + "\" " + NorS


#allow print statements to both write to the logfile and to sysout
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def dump_data(data,input_path):
    print("Dumping final pixel values to: \n\t%s" % 
            input_path.split(".tif")[0]+output_suffix()+".dat")
    f = open(input_path.split(".tif")[0]+output_suffix()+".dat","w")
    if DUMP_NODATA is True:
      for array in data:
        for datum in array:
          f.write(str(datum)+"\n")
      f.close()
    else:
      for datum in data[np.where(data != NODATA_VALUE)]:
        f.write(str(datum)+"\n")
      f.close()


def report_min_max(data,dtype,tabbed):

    if type(data) is list:
      arr = np.array([])
      for datum in data:
        arr=np.insert(arr,0,datum.flatten())
      data = arr

    if np.where(data != NODATA_VALUE)[0].size == 0:
      print("\t%s Max & Min:" % (dtype))
      print("\t\t% f" %(NODATA_VALUE))
      print("\t\t% f" %(NODATA_VALUE))
    elif tabbed is True:
      print("\t%s Max & Min:" % (dtype))
      print("\t\t% f" %(np.max(data[np.where(data != NODATA_VALUE)])))
      print("\t\t% f" %(np.min(data[np.where(data != NODATA_VALUE)])))
    else:
      print("-> %s Max & Min:" % (dtype))
      print("\t% f" %(np.max(data[np.where(data != NODATA_VALUE)])))
      print("\t% f" %(np.min(data[np.where(data != NODATA_VALUE)])))


#compute an estimate of the remaining run time
def eta(start,b,e,i,pixels_per_row):

    elapsed_s         = time.time()-start
    pixels_processed  = pixels_per_row * (i-b) 
    pixels_per_s      = pixels_processed/elapsed_s
    pixels_remaining  = pixels_per_row * (e-i)
    remaining_s       = pixels_remaining/pixels_per_s 

    if remaining_s < 61:
        return remaining_s, "s"
    elif remaining_s < 3601:
        return remaining_s/60.0, "m"
    elif remaining_s < 86401:
        return remaining_s/3600.0, "h"
    else:
        return remaining_s/86400.0, "d"


#report the progress as a percentage along with estimated remaining run time 
def report_progress(start,b,e,i,ncols,rows_per_process):

    spinner = "-\|/"
    if (i != b):
        remaining_time, unit = eta(start,b,e,i,ncols)
        print("\t\t %c %.1f%% (~%d%s remaining)" % 
          (spinner[i%4],((i-b)*ncols)/(rows_per_process*ncols)*100.0,
              remaining_time,unit),end="\r")
    else:
        print("\t\t %c %.1f%%" % 
          (spinner[i%4],((i-b)*ncols)/(rows_per_process*ncols)*100.0),end="\r")


#recursively prune away any undulating edges on the input .tif
def crop_edges(data,ncols,nrows,pruned):

    print("Checking edges for undulations...", end = "\r")

    spinner='\\|/-'

    #remove 1 pixel from each edge until we have no zeros
    i = 0
    for i in range(0,nrows,10):
        zero_cnt = np.count_nonzero(data[i:nrows-i,i:ncols-i] == 0)
        if zero_cnt == 0:
            if (i == 0):
                print("")
            break
        else:
            if (i == 0):
                 print("Cropping edges to remove undulations...", end = "\n")
            ind = (i/10)%4
            print("%c" % (spinner[int(ind)]), end = '\r')

    #try to expand the y-dimension and recover pixels
    j = 0
    for j in range(0,ncols,10):
        zero_cnt = np.count_nonzero(data[i:nrows-i,j:ncols-j] == 0)
        if zero_cnt == 0:
            break
        else:
            ind = ((i+j)/10)%4
            print("%c" % (spinner[int(ind)]), end = '\r')

    #try to expand the x-dimension and recover pixels
    k = 0
    for k in range(0,nrows,10):
        zero_cnt = np.count_nonzero(data[k:nrows-k,j:ncols-j] == 0)
        if zero_cnt == 0:
            break
        else:
            ind = ((i+j+k)/10)%4
            print("%c" % (spinner[int(ind)]), end = '\r')

    if (i+j+k != 0):
        print("--->Undulations detected, the input has been cropped!")
        print("--->initial shape = (%d x %d)" % (nrows,ncols))
        print("--->cropped shape = (%d x %d)" % (nrows-(2*k),ncols-(2*j)))
        print("--->User validation of new dimensions is suggested.")
    else:
        print("Edges are straight; no cropping required.")

    return data[k:nrows-k,j:ncols-j], ncols-(2*j), nrows-(2*k)


def splash():
    print("""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
               (   (( . : (    .)   ) :  )     *Dept. of Geology &           
                (   ( :  .  :    :  )  ))           **Dept. of Chemistry,    
                 ( ( ( (  .  :  . . ) )        University of South Florida   
                  ( ( : :  :  )   )  )                                       
                   ( :(   .   .  ) .'                                        
                    '. :(   :    )              /$$   /$$ /$$$$$$$$ /$$$$$$ 
                      (   :  . )  )           | $$$ | $$| $$_____//$$__  $$
                       ')   :   #@##          | $$$$| $$| $$     | $$  \__/
                      #',### \" #@  #@         | $$ $$ $$| $$$$$  | $$ /$$$$
                     #/ @'#~@#~/\   #         | $$  $$$$| $$__/  | $$|_  $$
                   ##  @@# @##@  `..@,        | $$\  $$$| $$     | $$  \ $$
                 @#/  #@#   _##     `\        | $$ \  $$| $$     |  $$$$$$/
               @##;  `#~._.' ##@      \_      |__/  \__/|__/      \______/ 
             .-#/           @#@#@--,_,--\                                 
            / `@#@..,     .~###'         `~.    Roughness Map Generator
          _/         `-.-' #@####@          \                         
       __/     &^^       ^#^##~##&&&   %     \_     *Nikola Rogic, Ph.D.
      /       && ^^      @#^##@#%%#@&&&&  ^    \     *Franco Villegas-Garin
    ~/         &&&    ^^^   ^^   &&&  %%% ^^^   `~_   **Guy Dayhoff II
 .-'   ^^    %%%. &&   ___^     &&   && &&   ^^    \                 
/akg ^^^ ___&&& X & && |n|   ^ ___ %____&& . ^^^^^ `~.        [ January 2022 ]
        |M|       ---- .  ___.|n| /\___\                                     
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """)


#create an informative suffix for the output file
def output_suffix():

    output_suffix = ""

    if REPORT_AS_LOG10 is True:
        output_suffix = "_log10"
        
    if ROUGHNESS_MODEL is None:
        output_suffix += "_backscatter"
    else:
        if MULTIPROCESSING is True:
            roughness_model = "M"
        else:
            roughness_model = "m"

        roughness_model += str(ROUGHNESS_MODEL)

        output_suffix += "_roughness_"
        output_suffix += str(max(WINDOWS))+"x" + \
            str(max(WINDOWS))+roughness_model

    if PINCH_INPUT is True:
        output_suffix += "_pinched"

    if INTENSITY_WEIGHTED is True:
        output_suffix += "_IW"
        if PINCH_WEIGHTS is True:
            output_suffix += "p"

    if RESCALE_OUTPUT is True:
        if COHERENT_OUTPUT is True:
            output_suffix += "_CRS"
        else:
            output_suffix += "_RS"

    if APPLY_GAUSSIAN_BLUR is True:
        output_suffix += "_GB"

    if CLUSTER_OUTPUT is True:
        output_suffix += "_clustered-"+str(CLASS_BOUNDARY)

    if len(WINDOWS) > 1:
        output_suffix += "_multipass"
        for window in WINDOWS:
            output_suffix += "-"+str(window)

    return output_suffix


#------------------------------------------------------------------------------
#  primary functions
#------------------------------------------------------------------------------

# slide a window over the image to construct the overlay
def slide(nrows, ncols, data, window, roughness_fn):
    roughness=np.full((nrows,ncols),NODATA_VALUE)
    start = time.time()

    margin = int(np.floor(window/2))

    for i in range(margin,nrows-margin):
        report_progress(start,margin,nrows-margin,i,ncols,nrows)
        for j in range(margin,ncols-margin):
            W = data[(i-margin):(i+margin+1),(j-margin):(j+margin+1)]
            if REPORT_AS_LOG10 is True:
                roughness[i,j] = np.log10(globals()[roughness_fn](W))
            else:
                roughness[i,j] = globals()[roughness_fn](W)
    return roughness


#process a given window, considering blank treatment first
def process_window(W,roughness,roughness_fn,i,j):

  #count number of blanks
  n_blanks = np.count_nonzero(W[:,:] == NODATA_VALUE)
  
  if AVOID_BLANKS is True and n_blanks != 0 and ROUGHNESS_MODEL != 9:
    roughness[i,j] = NODATA_VALUE 
  else:
    if WORKAROUND_BLANKS is True:
        W = W[np.where(W != NODATA_VALUE)] 
    if len(W) == 0:
        roughness[i,j] = NODATA_VALUE
    elif REPORT_AS_LOG10 is True:
        roughness[i,j] = np.log10(globals()[roughness_fn](W))
    else:
        roughness[i,j] = globals()[roughness_fn](W)
  
    if np.isinf(roughness[i,j]):
        roughness[i,j] = NODATA_VALUE


# slide multiple windows over the image to construct the overlay
def slide_mp(data, roughness, index, nrows, ncols, window, roughness_fn):
  
  #determine the scope of this process
  rows_per_process = int(np.ceil(nrows/mp.cpu_count()))
  
  b = rows_per_process*index
  e = rows_per_process*(index+1)

  #setup time tracking (only used in one process)
  start = time.time()

  margin = int(np.floor(window/2))

  #treat the special where we must decrement by 1 for marginal pixels
  if index == 0:
      b = margin
  elif index == (mp.cpu_count()-1):
      e = nrows-margin

  if (e>nrows-margin):
    e=nrows-margin

  for i in range(b,e):
      #report progress and estimated remaining time from a single process
      if index == mp.cpu_count()-2:
        report_progress(start,b,e,i,ncols,rows_per_process)

      #compute the roughness
      for j in range(margin,ncols-margin):
          W = data[(i-margin):(i+margin+1),(j-margin):(j+margin+1)]
          process_window(W,roughness,roughness_fn,i,j) 


#the gravy bby
def measure(nrows, ncols, data, window):

    roughness_fn = "m" + str(ROUGHNESS_MODEL)

    if MULTIPROCESSING is True:
       #set the roughness function
       print("\tComputing roughness (%s) using a %dx%d window...." % 
               (roughness_fn,window,window))

       start = time.time()

       #create shared memory arrays for the input and output
       shm_data = mp.RawArray('d',nrows*ncols)
       shm_roughness = mp.RawArray('d',nrows*ncols)
    
       #create numpy arrays to work on the shared memory arrays
       shm_data_np = np.frombuffer(shm_data).reshape((nrows,ncols))
       shm_roughness_np = np.frombuffer(shm_roughness).reshape((nrows,ncols))
    
       #initialize the shared memory arrays 
       np.copyto(shm_data_np,data)
       np.copyto(shm_roughness_np,np.full((nrows,ncols),NODATA_VALUE))
    
       #launch a series of processes to compute the roughness
       processes = list()
       for index in range(mp.cpu_count()):
           p = mp.Process(target=slide_mp,
                   args=(shm_data_np, shm_roughness_np, index,
                       nrows, ncols, window, roughness_fn))
           p.start()
           processes.append(p)
          
       #wait for each process to complete its task
       for index, p in enumerate(processes):
           p.join()

       #and we're done!
       runtime = time.time()-start

       print("                                                         ",
               end="\r")

       print("\t\t%s computed in %s                           " % 
               (roughness_fn.upper(),\
                       time.strftime("%Hh %Mm %Ss", time.gmtime(runtime))))

       return roughness_fn.upper(), shm_roughness_np
    else:
        #set the roughness function
        print("\tComputing roughness measure (%s)...." % (roughness_fn))
        
        #call it
        roughness = slide(nrows,ncols,data,window,roughness_fn)

        #and we're done!
        print("\n\t%s computed successfuly!" % (roughness_fn))
        return roughness_fn, roughness


#rescale data to be in the range [0,1]
def rescale_output(data):

    if type(data) is not list:
        max_val = np.max(data)
        min_val = np.min(data[np.where(data != NODATA_VALUE)])

        data[np.where(data != NODATA_VALUE)] -= min_val  
        data[np.where(data != NODATA_VALUE)] /= (max_val - min_val)
    else:
        min_val = 99999999999999.0
        max_val = -99999999999999.0

        for datum in data:
            if np.max(datum) > max_val:
                max_val = np.max(datum)
            if np.min(datum[np.where(datum != NODATA_VALUE)]) < min_val:
                min_val = np.min(datum[np.where(datum != NODATA_VALUE)])

        for datum in data:
            datum[np.where(datum != NODATA_VALUE)] -= min_val
            datum[np.where(datum != NODATA_VALUE)] /= (max_val - min_val)

    return data


#used for chimeric output: under construction
def minmax_consensus(data):
    data = rescale_output(data)
    threshold = (np.max(data)-np.min(data[np.where(data!=NODATA_VALUE)]))/2.0
    voters = np.shape(data)[0]
    consensus = np.array(data[0])
    consensus[:] = NODATA_VALUE;
    for i in range(0,voters):
        data[i] = rescale_output(data[i])

    spinner = "-\|/"
    for i, row in enumerate(data[0]):
        print(">> %s (%d/%d)" % (spinner[i%4],i,np.shape(data)[1]),end="\r")
        for j, ele in enumerate(row):
            over = 0
            under = 0

            for k in range(0,voters):
                if data[k][i][j] == NODATA_VALUE:
                    continue;
                elif data[k][i][j] <= threshold:
                    under += 1
                else:
                    over += 1

            if under == 0 and over == 0:
                consensus[i][j] = NODATA_VALUE
            elif over > under:
                consensus[i][j] = np.max(data[:,i,j])
            else:
                consensus[i][j] = np.min(data[:,i,j])

    return consensus


#consider the input path and yield a list containing the input file(s)
def construct_input_file_list(input_path):
    input_files = []

    if ".tif" in input_path:
        print("NFG will run on single input file. (%s)" % (input_path)) 
        input_files = [input_path]
    else:
        print("NFG will run on a directory of input files (%s)" % (input_path))
        tmp = [f for f in listdir(input_path) if isfile(join(input_path,f))]
        for target_file in tmp:
            if ".tif" in target_file:
                input_files.append(target_file)

    return input_files


def check_data_for_blanks(data):
    global AVOID_BLANKS
    global WORKAROUND_BLANKS
    
    if ROUGHNESS_MODEL == 9:
        AVOID_BLANKS = False
        WORKAROUND_BLANKS = False
        return

    if CONVERT_INPUT_TO_BS is True:
        blank_cnt = np.count_nonzero(data[:,:] <= 0)
    else:
        blank_cnt = np.count_nonzero(data[:,:] == NODATA_VALUE);

    if (blank_cnt != 0):
        print("\t\t%d potentially blank pixels detected." % (blank_cnt)) 

        if WORKAROUND_BLANKS is True:
            print("\t\tSliding windows will shrink to workout blanks.");
        else:
            print("\t\tBlanks will be avoided.");
            AVOID_BLANKS = True


incidence_angles = np.array(())

def load_incidence_angles(dataset,nrows,ncols):
    global incidence_angles

    incidence_angles = np.full((nrows,ncols),-1.0)

    xoffset, px_w, rot1, yoffset, rot2, px_h = dataset.GetGeoTransform() 

    crs = osr.SpatialReference()
    crs.ImportFromWkt(dataset.GetProjectionRef())
   
    crsGeo = osr.SpatialReference()
    crsGeo.ImportFromEPSG(4326)

    t2 = osr.CoordinateTransformation(crsGeo, crs)

    with open(ANGLE_FILE) as f:
      for line in f:
        if line[0] == "#":
            continue

        data=line.split()

        dd_lat=float(data[0])
        dd_lon=float(data[1])

        if float(data[3]) == NODATA_ANGLE:
            ang=float(90.0)*np.pi/180.0
        else:
            ang=float(data[3])*np.pi/180.0

        posX,posY,z=t2.TransformPoint(dd_lon,dd_lat)

        _X_ = (posY/px_h) - (yoffset/px_h) - ((rot2*posX)/(px_w*px_h)) 
        _X_ -= ((rot2*xoffset)/(px_w*px_h))/(((rot2*rot1)/(px_w*px_h)) + 1.0)
        _X_ = int(_X_)

        _Y_ = int(posX/px_w - xoffset/px_w - (rot1*_X_)/px_w)
       
        if (_X_%32 == 31):
            _X_ += 1
        if (_Y_%32 == 31):
            _Y_ += 1

        _X_UPPER = _X_ + 32
        _Y_UPPER = _Y_ + 32

        if _X_UPPER > nrows:
            _X_UPPER = nrows

        if _Y_UPPER > ncols:
            _Y_UPPER = ncols

        if (_X_ < nrows and _Y_ < ncols):
            incidence_angles[_X_:_X_UPPER,_Y_:_Y_UPPER] = ang #in radians


def extract_dataset(input_path,infile):
        input_file=input_path+"/"+infile

        if ".tif" in input_path:
            input_file=input_path

        print("\n>> Now processing input file: %s" % (input_file))

        #validate the input file
        if not exists(input_file):
          print("\tThe specified input file (%s) couldn't be found." 
                  % input_file)
          sys.exit()

        #register all available drivers for GDAL
        gdal.AllRegister()

        #try to open the file with GDAL
        dataset = gdal.Open(input_file, gdalconst.GA_ReadOnly) 
        if dataset is None:
          print("\tThe input file exists, however GDAL was unable to open it.")
          sys.exit()
        
        #grab the shape of the dataset, and the WGS84 georef data
        ncols = dataset.RasterXSize
        nrows = dataset.RasterYSize
       
        #grab the raw data
        print("\tPulling data from input file...")
        data = dataset.GetRasterBand(1).ReadAsArray(0, 0, ncols, nrows)
        
        check_data_for_blanks(data) 

        if ANGLE_FILE is not None:
            print("\tIncidence angle file detected. Loading data...")
            load_incidence_angles(dataset,nrows,ncols)

        return dataset, ncols, nrows, data


def convert_to_backscatter(data,use_og_eqn=False,dataset=None):
     if (use_og_eqn is False and ANGLE_FILE is None):
       print("\tConverting input, x, to backscatter, y.")
       print("\t\ti.e. y = 10log(DN^2-N)-42");
       print("\t\t     N = %f" % (IMAGE_NOISE_BIAS))
    
       data[np.where(data**2-IMAGE_NOISE_BIAS <= 0)] = 0
       data[np.where(data <= 0)] = NODATA_VALUE

       loaded = np.where(data != NODATA_VALUE)
       data[loaded] = (10.0*np.log10(data[loaded]**2-IMAGE_NOISE_BIAS))-42.0

       report_min_max(data,"Backscatter",True)
     elif (use_og_eqn is False):
       print("\tConverting input, x, to backscatter, y.")
       print("\t\ti.e. y = 10log(DN^2-N)+10log(sin(i_p))-42");
       print("\t\t     N = %f" % (IMAGE_NOISE_BIAS))
       print("\t\t     i_p taken from %s" % (ANGLE_FILE))

       data[np.where(data**2-IMAGE_NOISE_BIAS <= 0)] = 0
       data[np.where(data <= 0)] = NODATA_VALUE

       loaded = np.where(data != NODATA_VALUE)
       data[loaded] = (10.0*np.log10(data[loaded]**2-IMAGE_NOISE_BIAS)) + \
               10.0*np.log10(np.sin(incidence_angles[loaded]))-42.0

       report_min_max(data,"Backscatter",True)

     else:
       print("\tConverting input, x, to backscatter, y.")
       print("\t\ti.e. y = 20log(x^2)-42");

       data[np.where(data <= 0)] = NODATA_VALUE

       loaded = np.where(data != NODATA_VALUE)
       data[loaded] = (20.0*np.log10(data[loaded]**2))-42.0

       report_min_max(data,"Backscatter",True)

     return data


def pinch_data(data):
    print("\tPinching input (PINCH_FACTOR=%f)!" % (PINCH_FACTOR))
    
    sigma = np.std(data[np.where(data != NODATA_VALUE)]) 
    mu = np.mean(data[np.where(data != NODATA_VALUE)].flatten()) 
    
    print("\t\tmean(input) = %f" % (mu))
    print("\t\tstdv(input) = %f" % (sigma))
    
    new_min=mu-(PINCH_FACTOR*sigma)
    new_max=mu+(PINCH_FACTOR*sigma)
    
    report_min_max(data,"Raw input",True)

    data[np.where(data != NODATA_VALUE)] \
            = data[np.where(data!=NODATA_VALUE)].clip(new_min,new_max)
    
    report_min_max(data,"Pinched input",True)

    return data


def describe_distribution(weights):

    if type(weights) is not list:
        sigma = np.std(weights[np.where(weights!=NODATA_VALUE)]) 
        mu = np.mean(weights[np.where(weights!=NODATA_VALUE)].flatten()) 
    else:
        data = np.array([])
        for weight in weights:
            data=np.insert(data,0,weight.flatten())
        sigma = np.std(data[np.where(data != NODATA_VALUE)])
        mu = np.mean(data[np.where(data != NODATA_VALUE)])

    return sigma, mu


def intensity_weighted(results,weights,infile=None):
    
    if PINCH_WEIGHTS is True:
        sigma, mu = describe_distribution(weights)

        new_min=mu-(PINCH_FACTOR*sigma)
        new_max=mu+(PINCH_FACTOR*sigma)

        if COHERENT_OUTPUT is True:
          print("\n>> Now generating coherent pinched intensity weights\n\t"\
                + "PINCH_FACTOR=%f" % (PINCH_FACTOR))

          for weight in weights:
            weight[np.where(weight != NODATA_VALUE)] = \
              weight[np.where(weight != NODATA_VALUE)].clip(new_min,new_max)

        else:
          print("\n>> Now generating pinched intensity weights for" \
                + "%s\n\tPINCH_FACTOR=%f" % (infile, PINCH_FACTOR))
        
          weights[np.where(weights != NODATA_VALUE)] = \
                weights[np.where(weights != NODATA_VALUE)].\
                clip(new_min,new_max)

        print("\tmean(input) = %f" % (mu))
        print("\tstdv(input) = %f\n" % (sigma))
    
    #rescale to [0,1] to use as weights
    report_min_max(weights,"Intensity Weights",False)
    weights = rescale_output(weights)
    report_min_max(weights,"Rescaled Intensity Weights",False)
   
    if type(results) is not list:
        results[np.where(results != NODATA_VALUE)] *= \
            weights[np.where(results != NODATA_VALUE)]
    else:
        for index,result in enumerate(results):
            result[np.where(result != NODATA_VALUE)] *= \
                weights[index][np.where(result != NODATA_VALUE)]
    
    if PINCH_WEIGHTS is False:
        report_min_max(results,"Intensity Weighted",False)
    else:
        report_min_max(results,
                    "Intensity Weighted w/ Pinched Weights)",False)

    return results


def process_input(input_files, input_path):
    datasets=[]
    contribs=[]
    results=[]
    weights=[]

    #loop over each requested window size
    for count, window in enumerate(WINDOWS): 

      #loop over each input file
      for index, infile in enumerate(input_files):
        dataset, ncols, nrows, data = extract_dataset(input_path,infile) 

        #init matrices, output=results/contributions
        result = np.zeros(shape=(nrows,ncols))
        contrib = np.zeros(shape=(nrows,ncols))
      
        #convert the input (which should be amplitudes) into backscatter p.r.n.
        if CONVERT_INPUT_TO_BS is True:
            data = convert_to_backscatter(data,USE_OG_BS_EQN,dataset)      

        #constrain the input data range p.r.n.
        if PINCH_INPUT is True:
            data = pinch_data(data)    

        #meASuRe dE rouGHneSS p.R.n
        if ROUGHNESS_MODEL != None:
            with np.errstate(divide='ignore'):
                roughness_model, roughness = measure(nrows,ncols, \
                        data,window)
                report_min_max(roughness,roughness_model,True)
        else:
            roughness = data

        #capture inverted input data to use for intensity weighting p.r.n.
        if INTENSITY_WEIGHTED is True:
            weight = -1.0*data;
            #correct the inverted NODATA_VALUES
            weight[np.where(data == NODATA_VALUE)] = NODATA_VALUE;
        else:
            weight = None

        result[np.where(roughness != NODATA_VALUE)] += \
            roughness[np.where(roughness != NODATA_VALUE)]

        #track the number of contributions for proper averaging
        contrib[np.where(roughness != NODATA_VALUE)] += 1

        #collect the dataset, results, and contributions
        datasets.append(dataset)
        contribs.append(contrib)
        results.append(result)
        weights.append(weight)
        
        #dereference the dataset so you don't get a file full of zeros 
        roughness = None

    #reduce to average over windows for multipass runs
    if len(WINDOWS) > 1 and ROUGHNESS_MODEL != None:
        print("\n>> Averaging results over %d window sizes." % 
                (len(WINDOWS)))

    #TODO: correct this code for multi-pass cases
    for index, result in enumerate(results):
        result[np.where(contribs[index] != 0)] /= \
                contribs[index][np.where(contribs[index] != 0)]

        result[np.where(contribs[index] == 0)] = NODATA_VALUE

    return results, contribs, weights, datasets


def write_output(dataset,output,input_path):

    ncols = dataset.RasterXSize
    nrows = dataset.RasterYSize

    #grab the georeference data
    georef = dataset.GetGeoTransform() 

    #i'm not 100% clear on what we are doing here...but ¯\_(ツ)_/¯
    x_xyH = np.min(georef[0] + georef[1]*np.arange(0,ncols,1))
    y_xyH = np.max(georef[3] + georef[5]*np.arange(0,nrows,1))


    if SPLIT_OUTPUT is False:
      #create a new dataset and write to disk
      print("\nWriting final output to:\n\t %s\n" % 
              (input_path.split(".tif")[0]+output_suffix()+".tif"))

      new_dataset = dataset.GetDriver().Create(
              input_path.split(".tif")[0]+output_suffix()+".tif",
              ncols,nrows,1,gdalconst.GDT_Float32)
      
      new_dataset.GetRasterBand(1).WriteArray(output,0,0)
      new_dataset.GetRasterBand(1).SetNoDataValue(NODATA_VALUE)
      new_dataset.SetGeoTransform((x_xyH,georef[1],0,y_xyH,0,georef[5]))
      new_dataset.SetProjection(dataset.GetProjection())

      #dereference the dataset so you don't get a file full of zeros 
      output = None
      new_dataset = None
    else:
        print("\nSplitting output using a threshold of %f\n" % 
                (SPLIT_THRESHOLD))

        output_above = np.copy(output)
        output_below = np.copy(output)

        output_above[np.where(output <= SPLIT_THRESHOLD)] = NODATA_VALUE
        output_below[np.where(output >= SPLIT_THRESHOLD)] = NODATA_VALUE

        #create a new dataset and write to disk
        print("\nWriting split output to:\n\t %s\n" %
                (input_path.split(".tif")[0]+output_suffix()+".SPLIT.tif"))

        anew_dataset = dataset.GetDriver().Create(
                input_path.split(".tif")[0]+output_suffix()+".SPLIT.tif",
                ncols,nrows,1,gdalconst.GDT_Float32)

        anew_dataset.GetRasterBand(1).WriteArray(output_above,0,0)
        anew_dataset.GetRasterBand(1).SetNoDataValue(NODATA_VALUE)
        anew_dataset.SetGeoTransform((x_xyH,georef[1],0,y_xyH,0,georef[5]))
        anew_dataset.SetProjection(dataset.GetProjection())

        #create a new dataset and write to disk
        print("\nWriting split output to:\n\t %s\n" %
                (input_path.split(".tif")[0]+output_suffix()+".split.tif"))

        bnew_dataset = dataset.GetDriver().Create(
                input_path.split(".tif")[0]+output_suffix()+".split.tif",
                ncols,nrows,1,gdalconst.GDT_Float32)

        bnew_dataset.GetRasterBand(1).WriteArray(output_below,0,0)
        bnew_dataset.GetRasterBand(1).SetNoDataValue(NODATA_VALUE)
        bnew_dataset.SetGeoTransform((x_xyH,georef[1],0,y_xyH,0,georef[5]))
        bnew_dataset.SetProjection(dataset.GetProjection())

        #dereference the dataset so you don't get a file full of zeros
        output = None

        output_above = None
        output_below = None
        anew_dataset = None
        bnew_dataset = None


#serve it up hot & fresh
def workup_results(input_path,output,dataset):

    ncols = dataset.RasterXSize
    nrows = dataset.RasterYSize

    if (RESCALE_OUTPUT is True and COHERENT_OUTPUT is False) \
        or CLUSTER_OUTPUT is True:
            report_min_max(output,"Intermediate Output ("+input_path+")",False)

    if RESCALE_OUTPUT is True and COHERENT_OUTPUT is False:
        output = rescale_output(output)
        print("\n**Output has been rescaled into the range [0,1]\n")

    if CLUSTER_OUTPUT is True:
        output[np.where(output >= CLASS_BOUNDARY)] = \
            np.ceil(output[np.where(output >= CLASS_BOUNDARY)])
        output[np.where(output < CLASS_BOUNDARY)] = \
            np.floor(output[np.where(output < CLASS_BOUNDARY)])
        print("The final output has been been clustered into 2 classes," \
                +" i.e. 0=smooth, 1=rough")

    report_min_max(output,"Final Output ("+input_path+")",False)

    #assign the marginal pixels the NODATA value
    if ROUGHNESS_MODEL is not None:
      margin = int(np.floor(max(WINDOWS)/2))
      output[:margin,:] = NODATA_VALUE
      output[nrows-margin:,:] = NODATA_VALUE
      output[:,:margin] = NODATA_VALUE
      output[:,ncols-margin:] = NODATA_VALUE
    
    nodatas=len(np.where(output[:,:] == NODATA_VALUE)[0])

    print("\nTotal pixels in roughness map: %s" % ("{:,}".\
            format(ncols*nrows)))

    print("\nTotal blanks in roughness map: %s (%d%%)\n" % 
            ("{:,}".format(nodatas),nodatas/(ncols*nrows)*100.0))

    #dump all pixel values to a single-column file
    if DUMP_DATA is True:
        dump_data(output,input_path)
  
    #write the new map
    write_output(dataset,output,input_path)


#the meat and potatoes
def main(input_path):

    input_files = construct_input_file_list(input_path)
   
    global filter
    if APPLY_GAUSSIAN_BLUR is True:
        center = np.floor(WINDOWS[0]/2)
        for i in range(WINDOWS[0]):
          I=i-center
          for j in range(WINDOWS[0]):
            J=j-center
            filter[i,j] = k0(I,J,BLUR_FACTOR)

    results,contribs,weights,datasets = process_input(input_files,input_path)

    #apply intensity weighting p.r.n.
    if INTENSITY_WEIGHTED is True:
      if COHERENT_OUTPUT is True:
        #treat the entire input space as a single input
        results = intensity_weighted(results,weights)
      else:
        #treat each input individually
        for index,infile in enumerate(input_files):
          results[index] = intensity_weighted(results[index], \
                  weights[index],input_path+"/"+infile)

    for i in range(78):
        print("~",end="")
    print("~")
         
    #produce a single output averaged over all inputs
    if AVERAGE_INPUTS is True:
      average = results[0]
      if len(input_files) > 1:
          print("\n>> Averaging results yielded by %d inputs" % 
                  (len(input_files)))
          for index, infile in enumerate(input_files): 
              if index == 0:
                  continue
              average += results[index]
          average /= len(input_files)
      workup_results(input_path,average,datasets[0])
      results[0] = None
    else:
      if RESCALE_OUTPUT is True and COHERENT_OUTPUT is True:
        results = rescale_output(results)
        print("\n**All results have been coherently rescaled " \
                +"into the range [0,1]\n")

      for index, infile in enumerate(input_files):
        workup_results(input_path+"/"+infile,results[index],datasets[index])
        results[index] = None
    
    return input_path


#------------------------------------------------------------------------------
# context dependent execution
#------------------------------------------------------------------------------

if __name__ == "__main__":

    #set a temporary log file
    f = open('nfg_logfile.tmp', 'w')
    backup = sys.stdout
    sys.stdout = Tee(sys.stdout, f)

    splash()

    try:
        ROUGHNESS_MODEL = int(sys.argv[3])
        windows = sys.argv[2].split(",")
        WINDOWS = []
        for window in windows:
          WINDOWS.append(int(window))
        input_path = main(sys.argv[1]) 
    except IndexError:
        #INPUT_PATH is defined as a parameter at the top of the file
        input_path = main(INPUT_PATH) 

    #perform housekeeping on the log file
    system("grep -v 'progress' nfg_logfile.tmp > %s" % 
            (input_path.split(".tif")[0]+output_suffix()+".log"))
    remove("nfg_logfile.tmp")
