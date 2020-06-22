#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 19:21:09 2020

@author: bfemenia
"""


# %%  IMPORT SECTION
#-------------------
import pandas as pd 
import shutil


# %% MAIN CODE
#-------------
def select_ip_DC_files(lmin=0, lmax=10, bmax=20, op_name='batch_DC.txt', count=False,
                       ip_path='/home/TFM/IP_files_DC/', op_path='/home/TFM/'):
    '''
    Selects the Dias Catalog files corresponding to clusters with l <= lmax
    and |b| <= 20.
    
    The file names are put into file opname and these will be reaad by the 
    MapREduce in the Hadoop cluster as a typical batch job.
    
    It also copies the ip_file_DC_? into the folder where MapReduce will be
    launched.
    
    CAUTION: make sure you delete your old batch_DC.txt!!!
    
    NOTICE: although MapRed will only be ran on the files in opname batch file
            it is convenient to remove the unneeded ip_file_DC to avoid 
            confusions.  

    Parameters
    ----------
    lmax : TYPE, numeric.  
        Use files with ceter up to lmax. The default is 40.
        
    bmax : TYPE, numeric
        Use files whose centers have |b| <= bmax. The default is 20.
        
    ip_path : TYPE, string
        Path where ALL ip_file_DC_*.csv files are located. 
        The default is '/home/TFM/IP_files_DC/'.
        
    op_path : TYPE, string
        Path where the batch file and the ip_file_DC_* to be considered will
        be saved. ALL ip_file_DC_*.csv files are located. 
        
    op_name : TYPE, string
        O/P file name including path. The default is 'batch_DC.txt'.

    Returns
    -------
    None. O/P is directly stored in opname file.

    '''
    
    df = pd.DataFrame()
    
    for i in range(1021):
        
        ip_file ='ip_file_DC_'+str(i).zfill(5)+'.csv'
        df_tmp  = pd.read_csv(ip_path + ip_file)
        
        if ( (df_tmp['l'][0] >= lmin) and (df_tmp['l'][0] < lmax) and 
            (abs(df_tmp['b'][0]) <= bmax) ):
            
            if count:
                df= df.append(df_tmp, ignore_index=True)
                
            else:
                with open(op_path+ op_name,'a') as f:
                    #Writing selected filename into batch file
                    f.write(ip_file+'\n')
                    #Copy parameter csv file into folder where batch lives
                    shutil.copyfile(ip_path + ip_file,
                                    op_path + ip_file)

    if count:
        print(f'\n\t #Files in DC within l-range= [{ lmin},{ lmax}[: \t\t {df.shape[0]}')
                
# %%
# select_ip_DC_files(count=True)
select_ip_DC_files( 0.0,  5.5, 20, 'batch_DC_l_00_l_05.txt', count=False)
select_ip_DC_files( 5.5, 14.7, 20, 'batch_DC_l_05_l_14.txt', count=False)
select_ip_DC_files(14.7, 28.0, 20, 'batch_DC_l_14_l_28.txt', count=False)
select_ip_DC_files(28.0, 68.4, 20, 'batch_DC_l_28_l_68.txt', count=False)