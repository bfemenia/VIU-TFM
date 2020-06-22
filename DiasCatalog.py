#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 14:57:32 2018

@author: rafa
"""

from astroquery.vizier import Vizier
from astropy.coordinates import Angle, Longitude, Latitude, SkyCoord
from astropy.table import Column, Table
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt

#To time code execution
import time

def parse_radec(ra, dec):
    tmp    = ra.split()
    ra_hms = tmp[0]+'h'+tmp[1]+'m'+tmp[2]+'s'
    
    tmp    = dec.split()
    dec_deg= tmp[0]+'d'+tmp[1]+'m'+tmp[2]+'s' 
    
    return ra_hms, dec_deg


def load_full_DiasCatalog(file='/home/bfemenia/TFM/DiasCatalog.pkl'):

    table=0
    
    try:
        with open(file,'rb') as f:
            table=pickle.load(f)
            
    except IOError as e:
        print(f'Error loading file with catalog. Error is{e}')
    
    except:
        print(f'Error reading file with catalog')
         
    finally:
        if len(table) <= 2:
            print('No catalog loaded!')
        else:
            print(f'\tCatalog in {file} loaded succesfully')
             
    return(table)


class DiasCatalog(object):
    
# %%
    def __init__(self, *attributes, ra_0=0, dec_0=0,r_0=0.5,entire=False,galactic=False):
        """
        Bruno Femenia Castella:
            
            in order to speed up calcularions it does not make sense to scan 
            the whole catalog but only those clusters found within 1.5*r_0 [degs]
            around coordinates [ra_0, dec_0]
        """
        self.attributes = attributes
        Vizier.ROW_LIMIT = -1

        if entire:
            dias_catalog = Vizier.get_catalogs('B/ocl/clusters')
        else:
            center_coord = SkyCoord(ra=ra_0, dec=dec_0, unit=(u.deg, u.deg), frame='icrs')
            dias_catalog = Vizier.query_region(center_coord, radius=1.5*r_0*u.deg, catalog='B/ocl/clusters')
 
    
        if attributes == ():
            self.table = dias_catalog['B/ocl/clusters']
        else:
            self.table = dias_catalog['B/ocl/clusters'][attributes]
            
            
        if galactic:
            nel   = len(self.table)
            Col_l = Column(name='l', unit=u.deg, dtype=np.float32, 
                           description='Galactic longitude', length=nel)
            Col_b = Column(name='b', unit=u.deg, dtype=np.float32, 
                           description='Galactic latitude',  length=nel)
            for i,row in enumerate(self.table):
                ra       = Angle(row['RAJ2000'], unit=u.hourangle)
                dec      = Angle(row['DEJ2000'], unit=u.deg)
                cl_radec = SkyCoord(ra, dec, frame='icrs')
                # ra_hms, dec_deg = parse_radec(row['RAJ2000'], row['DEJ2000'])
                # cl_radec=SkyCoord(ra_hms, dec_deg,frame='icrs')
                Col_l[i] = cl_radec.galactic.l.deg
                Col_b[i] = cl_radec.galactic.b.deg
                
            self.table.add_columns([Col_l, Col_b])   
       
        
 # %%
    def get_ra_dec(self, cluster):
        """Method to get right ascension and declination from a cluster
            
        Parameters
        --------
        cluster: Name of the cluster to get its position
        Returns
        --------
        SkyCoord object containing ra and dec in degrees
        """
        try:
            row = self.table[np.where(self.table['Cluster'] == cluster)]
            ra = Angle(row['RAJ2000'], unit=u.hourangle)
            dec = Angle(row['DEJ2000'], unit=u.deg)
            return([float(ra.deg), float(dec.deg)])
        except:
            print('Cluster does not exist')
            

# %%
    def get_cluster_names(self):
        """Method to get the name of the clusters in the catalog
        Returns
        --------
        print out name of the clusters in the catalog
        """
        print(self.table['Cluster'])
        
        
# %%
    def get_diam(self, cluster):
        """Method to get right ascension and declination from a cluster
            
        Parameters
        --------
        cluster: Name of the cluster to get its diameter
        Returns
        --------
        SkyCoord object containing ra and dec in degrees
        """
        try:
            row = self.table[np.where(self.table['Cluster'] == cluster)]
            diam = float(row['Diam'])*u.arcmin
            return(diam.to('deg').value)
        except:
            print('Cluster does not exist')
            
    
# %%    
    def get_cluster_coordinates(self):
        ra = Angle(self.table['RAJ2000'], unit=u.hourangle)
        ra = ra.deg
        ra = Longitude(ra, unit=u.deg)
        dec = Latitude(self.table['DEJ2000'], unit=u.deg )
        return(SkyCoord(ra, dec))
    
    
# %%    
    def plot_all(self):
        coordinates = self.get_cluster_coordinates()
        ra = coordinates.ra
        dec = coordinates.dec
        fig, ax = plt.subplots()
        ax.scatter(ra, dec, s=0.5)
        #for i, txt in enumerate(np.array(self.table['Cluster'], dtype='S')):
        #    ax.annotate(txt, (ra[i].value, dec[i].value))
        plt.show()
        
        
# %%    
    def get_clusters(self, center, r):
        # print('\t\tDentro de DiasCatalog.get_clusters() method.')

        # tA= time.time()
        clusters = []
        ra_center = Longitude(center[0], unit=u.deg)
        dec_center = Latitude(center[1], unit=u.deg)
        # print('\t\tDentro de DiasCatalog.get_clusters() method.')
        center = SkyCoord(ra_center, dec_center)
        # tB= time.time()
        # print(f'\t\t\t >> Seconds before internal loop: {tB-tA}')
        
        # tC= time.time()
        # print(f"\t\t\t >> Starting internal loop with #clusters in Dias Catalog: {self.table['Cluster'].shape[0]}")
        for i in range(self.table['Cluster'].shape[0]):
            tD= time.time()
            ra = Angle(self.table[i]['RAJ2000'], unit=u.hourangle)
            ra = ra.deg
            ra = Longitude(ra, unit=u.deg)
            dec = Latitude(self.table[i]['DEJ2000'], unit=u.deg)
            cluster = SkyCoord(ra, dec)
            diam = float(self.table[i]['Diam'])*u.arcmin
            if(np.isnan(diam)):
                diam = 0.05
            else:
                diam = diam.to('deg').value
            if(cluster.separation(center).value <= r):
                clusters.append((self.table['Cluster'][i], cluster, diam))
                
        #     if i==0:
        #         print(f"\t\t\t\t 1st iter in internal loop takes: {time.time()-tD} seconds")

        # print(f"\t\t\t >> Done with internal loop. Seconds to scan all clusters: {time.time()-tC} ")
        # print(f"\t\t >> Done with DiasCatalog.get_clusters() in  {time.time()-tA} seconds")



        return(clusters)
    
    
# %%    
    def get_rate(self, center, r, clusters):
        '''
        Method to get a metric between 0 and 1 to measure the success of our 
        algorithm.
        
        Parameters
        ---------
        center : List
                 Center of the sky
        r : float
            Radius of the sky
        
        cluster: List
                 Center point of a cluster detected
                 
        Returns
        -------
        rate: float
              Metric between 0 and 1 that measures how good our algorithm worked
        '''
        num_cum = int(0)
        matches = int(0)
        rate = 0.0
        ra = Longitude(center[0], unit=u.deg)
        dec = Latitude(center[1], unit=u.deg)
        center = SkyCoord(ra, dec)
#        ra_cluster = Longitude(cluster[0], unit=u.deg)
#        dec_cluster = Latitude(cluster[1], unit=u.deg)
#        cluster = SkyCoord(ra_cluster, dec_cluster)
        for i in range(self.table['Cluster'].shape[0]):
            ra_dias = Angle(self.table[i]['RAJ2000'], unit=u.hourangle)
            ra_dias = ra_dias.deg
            ra_dias = Longitude(ra_dias, unit=u.deg)
            dec_dias = Latitude(self.table[i]['DEJ2000'], unit=u.deg)
            cluster_dias = SkyCoord(ra_dias, dec_dias)
            diam = float(self.table[i]['Diam'])*u.arcmin
            if(np.isnan(diam)):
                diam = 0.05
            else:
                diam = diam.to('deg').value
            if(cluster_dias.separation(center).value <= r):
                num_cum += 1
                for cluster in clusters:
                    ra_cluster = Longitude(cluster[0], unit=u.deg)
                    dec_cluster = Latitude(cluster[1], unit=u.deg)
                    cluster = SkyCoord(ra_cluster, dec_cluster)
                    if(cluster_dias.separation(cluster).value <= diam/2.0):
                        matches += 1
                        break
        try:
            rate = matches/num_cum
            print('Number of cluster in the sky %5d' % num_cum)
            print('Number of matches: %5d' % matches)
        except:
            print('No star cluster detected in Dias catalog in the part of\n the sky specified')
        return(rate)
    
        
# %%        
    def get_match(self, center, r, clusters_detected):
        # print('\t\tDentro de DiasCatalog.get_match() method.')

        # tA= time.time()
        existing_clusters = self.get_clusters(center, r)
        # tB= time.time()
        # print(f'\t\t\t >> Seconds to run .get_clusters(): {tB-tA}')

        tot_matches = []
        for cluster in clusters_detected:
            ra_cluster_detected = Longitude(cluster[0], unit=u.deg)
            dec_cluster_detected = Latitude(cluster[1], unit=u.deg)
            cluster_detected = SkyCoord(ra_cluster_detected, dec_cluster_detected)
            l_tmp = [match for match in existing_clusters if match[1].separation(cluster_detected).value <= match[2]/2.0]
            tot_matches = list(set(tot_matches + l_tmp))

        # tC= time.time()
        # print(f'\t\t\t >> Seconds to run internal loop  : {tC-tB}')
        # print(f'\t\t\t >> Seconds to run .get_match()   =>{tC-tA}')

        return tot_matches
            
# %%         
#dias_catalog = DiasCatalog()
#center = [93.6084, 14.6927]
#r = 5.0
#cumulos_detectados = [[89.5583, 16.55944], [92.1, 13.965]]
#matches = dias_catalog.get_match(center, r, cumulos_detectados)
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
        
        
        