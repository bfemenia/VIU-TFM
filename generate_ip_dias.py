#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11May 14 20:17:23 2020

@author: bfemenia
"""

# %%  IMPORT SECTION
#-------------------
import pandas as pd

from astropy import units as u
from astropy.coordinates import Angle, Distance, Latitude, Longitude, SkyCoord
from reducerTFM import DiasCatalog as DiasCatalog


# %% MAIN CODE
#-------------
def generate_ip_dias(l_max=360, r_max=1.5, n_min=3, path='home/bfemenia/TFM/IP_files_DC'):
    """
    Bruno Femenia Castella
    
    This routine generates the selection of ip files over the Galactic plane
    containing at least 3 clusters in Dias catalog and not beyond r_max degs of
    size. Each of these sky patches witll be analyzed to derived eps=eps(l,b) 
    and min_pts=min_pets(l,b) which will guide the search for clusters in a 
    subsequent exploration of Gaia data.
    
    Based on these files we evaluate the (eps,Nmin) to be used over different 
    regions of whole Galactic plane to find new candidates with DBSCAN.

    Returns
    -------
    None.

    """
    
    full_dc = DiasCatalog('Cluster', 'RAJ2000','DEJ2000', 'l', 'b', 'Class',
                          'Diam', entire=True)
    gal_disk= full_dc.table[ (abs(full_dc.table['b']) < 20) & (full_dc.table['l'] < l_max) ]
   
    patches=[]
    
    for i, this_cluster in enumerate(gal_disk):   #Iterating over ALL clusters 
                                                  #in selected region |b|< 20 and l < l_max 
        c1       = SkyCoord(this_cluster['l']*u.deg, 
                            this_cluster['b']*u.deg, 
                            frame='galactic')
        clusters = gal_disk[ (abs(gal_disk['b']-this_cluster['b']) <= r_max)]
        
        
        if len(clusters) < n_min:           #Requesting a minimum number of clusters in this patch!!                                
            continue
        
        candidates=[{'name':this_cluster['Cluster'], 'ang_sep':0., 
                     'l':this_cluster['l'], 'b':this_cluster['b']}]
        
        for cluster in clusters:
            c2      = SkyCoord(cluster['l']*u.deg, cluster['b']*u.deg, frame='galactic')  
            ang_sep = c1.separation(c2).deg

            if (ang_sep <= r_max) and (ang_sep > 0):
                new = dict({'name':cluster['Cluster'], 'ang_sep':ang_sep,   
                            'l':cluster['l'], 'b':cluster['b']})
                candidates.append(new)                      
        
        if len(candidates) >= n_min:
            df = pd.DataFrame.from_records(candidates)   
            df.sort_values('ang_sep', inplace=True)
            patch = df[0:n_min].mean()                          #Use only n_min clusters at a time!!
            patches.append(patch)

            
    patches_df = pd.DataFrame.from_records(patches)
    patches_df.drop_duplicates(subset=['l','b'], inplace=True)  #Removing duplicates
    patches_df.reset_index(drop=True, inplace=True)
    
    #Adding rest of fields as defined by Rafa in his Thesis. Then save csv.
    #----------------------------------------------------------------------
    op_dict={'ra':[0.],
             'dec':[0.],
             'l':[0.],
             'b':[0.],
             'r':[0.],
             'err_pos':[100],
             'g_lim':[18.0],
             'norm':[None],
             'sample':[1.],
             'dim3':[None],
             'distance':['euclidean'],
             'eps_min':[0.008],
             'eps_max':[0.03],
             'eps_num':[100],
             'min_pts_min':[10],
             'min_pts_max':[100],
             'min_pts_num':[91]}
    op_df=pd.DataFrame(data= op_dict)
    
    for index, row in patches_df.iterrows():
        fn = 'ip_file_DC_'+str(index).zfill(5)+'.csv'
        c1 = SkyCoord(row['l']*u.deg, row['b']*u.deg, frame='galactic')
        
        op_df[ 'ra']= c1.icrs.ra.deg
        op_df['dec']= c1.icrs.dec.deg
        op_df[  'l']= row['l']
        op_df[  'b']= row['b']
        op_df[  'r']= row['ang_sep']
        
        op_df.to_csv(fn, index=False, header=True)
        
    return patches_df
