#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 11:23:54 2019

@author: rafa

Heavily modified by Bruno Femenía Castellá (bruno.femenia@gmail.com) to optiimize 
execution time and arragned to be exectued as apReduce process within a Hadoop cluster
"""

# %% ############   IMPORT Section   ############
# %%

# Main basic modules: numpy, pands and then also sys and datetime
import sys
import numpy as np
import pandas as pd
import datetime            # used for O/P filenames
import time                # used to time code execution
import pickle              # used to load Dias catalog
import math
    
#Sklearn algorithm
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

#To suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Third-party dependencies
from astropy import units as u
from astropy.coordinates import Angle, Distance, Latitude, Longitude, SkyCoord
from astropy.table import Column, Table
from astroquery.gaia import Gaia


# %%
def load_full_DiasCatalog(file='/home/TFM/DiasCatalog.pkl', verbose=False):
    '''
    Bruno Femenia Castella:
        
    Simple coad to load a pre-downloaded full Dias Catalog from Vizier
    using the DiasCatalog class. The download of the full catalog from
    Vizier exectued using a lsightly modified version of what was written by
    Rafa Lopez in his TFM.
       
    Because of problems with importing astroquery.vizier when running the 
    code in a MapRed framweork within a Hadoop cluster, this approach was needed.
       
    As a side effect, I tested that the time to identify the Dias catalog
    clusters in an specific part of the sky (region in Rafa's TFM) is now 
    6.5 times faster than the same specific region query to Vizier. BTW, this
    later approac is the one that helped to speed the code a factor 10 wrt 
    the original Rafa's TFM code
    '''
    
    table=0
  
    try:
        with open(file,'rb') as f:
            table=pickle.load(f)
            
    except IOError as e:
        print(f'Error loading file with catalog. Error is{e}')
    
    except:
        print('Error reading file with catalog')
         
    finally:
        if len(table) <= 2:
            print('No catalog loaded!')
        else:
            if verbose:
                print(f'\tCatalog in {file} loaded succesfully')
             
    return(table)



# %% ############   CLASSES DEFINITION Section   ############
# %%

# %%     ********   Dias Catalog class   ********
class DiasCatalog(object):
    
# %%
    def __init__(self, *attributes, l0=0, b0=0, radius=0.5, 
                 entire=False, DiasFile='/home/TFM/DiasCatalog.pkl'):
        """
        Bruno Femenia Castella:
            
            in order to speed up calcularions:

            1/ it does not make sense to scan the whole catalog but only those clusters
               found within 1.5*radius [degs] around coordinates [ra_0, dec_0]

            2/ It turns out astroquery.Vizier triggers an error in the MapRed in Hadoop.
               Rewritting this module so catalog is in a pkl
        """

   
        self.attributes = attributes

        table = load_full_DiasCatalog(file=DiasFile, verbose=False)             #Full Dias catalog here    
        if attributes != ():
            table = table[attributes]

        if entire:
            self.table = table
            
        else:
            #Computing distance of ALL Dias Catalog clusters to (l0, b0)
            Delta_l = ( (table.as_array(names=['l']).data).astype('float64') - l0) *(np.pi/180.) #conversion to rads
            b2      =   (table.as_array(names=['b']).data).astype('float64')       *(np.pi/180.) #conversion to rads
            cb2_cb1 = np.cos(b2)*math.cos(math.radians(b0))
            sb2_sb1 = np.sin(b2)*math.sin(math.radians(b0))
            dist    = np.arccos( np.cos(Delta_l)*cb2_cb1 + sb2_sb1 ) * 180./np.pi                #conversion to degs
            
            #Selecting only those with dist <= 1.5 * radius
            self.table = table[dist <= 1.5*radius]
               
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
    def get_clusters(self, center, r):
        clusters   = []
        ra_center  = Longitude(center[0], unit=u.deg)
        dec_center = Latitude(center[1], unit=u.deg)
        center     = SkyCoord(ra_center, dec_center)

        for i in range(self.table['Cluster'].shape[0]):
            ra      = Angle(self.table[i]['RAJ2000'], unit=u.hourangle)
            ra      = ra.deg
            ra      = Longitude(ra, unit=u.deg)
            dec     = Latitude(self.table[i]['DEJ2000'], unit=u.deg)
            cluster = SkyCoord(ra, dec, frame='icrs')
            diam    = float(self.table[i]['Diam'])*u.arcmin

            if(np.isnan(diam)):
                diam = 0.05
            else:
                diam = diam.to('deg').value
                
            if(cluster.separation(center).value <= r):
                clusters.append((self.table['Cluster'][i], cluster, diam))

        return(clusters)
    
# %%     
    def get_match(self, center, r, clusters_detected):
        existing_clusters = self.get_clusters(center, r)

        tot_matches = []
        for cluster in clusters_detected:
            ra_cluster_detected = Longitude(cluster[0], unit=u.deg)
            dec_cluster_detected = Latitude(cluster[1], unit=u.deg)
            cluster_detected = SkyCoord(ra_cluster_detected, dec_cluster_detected)
            l_tmp = [match for match in existing_clusters if match[1].separation(cluster_detected).value <= match[2]/2.0]
            tot_matches = tot_matches + l_tmp
            
        #building a Pandas DataFrame to easily remove duplicates by Name
        df = pd.DataFrame.from_records(tot_matches, columns=['Name','SkyCoord','Diam'])
        df.drop_duplicates(subset=['Name'], inplace=True)
        
        #Re-assembling tot_matches as a list of tuples
        tot_matches = []
        for row in df.iterrows():
            tmp = tuple(row[1])
            tot_matches.append(tmp)
 
        return tot_matches
 
    
# %%     ********   Gaia Data query class   ********
class GaiaData(object):
 
# %%    
    def __init__(self, attributes, coordsys = 'ICRS'):
        """Constructs an object representing an ADQL query
            
        Parameters
        --------
        attributes: List of fields to query (projection)
        point: point in the sky to query
        extent: length of the box centered in the point
        """
        self.attributes = attributes
        self.coordsys = coordsys

# %%    
    def astrometric_query_box(self, point, length, err_pos, g_lim):
        """Construct a query of a sky box centered in point specified when creating this object

        Parameters:
        ----------
        point: Point of the vertex of the box
        extent: lenght of the arms of the bos
        
        Returns
        --------
        ADQL Query
        data retrieve from the query
        
        """
        query = "SELECT %s"%', '.join(self.attributes)
        query += " FROM gaiadr2.gaia_source"
        query += " WHERE CONTAINS(POINT('%s', ra, dec), BOX('%s', %s, %s, %s))=1"%(self.coordsys, 
                                 self.coordsys, ', '.join([str(n) for n in point]), str(length), str(length))
        query +=" AND parallax IS NOT NULL"
        query +=" AND parallax >= 0.0"
        query +=" AND ra IS NOT NULL"
        query +=" AND dec IS NOT NULL"
        query +=" AND pmra IS NOT NULL"
        query +=" AND pmdec IS NOT NULL"
        query +=" AND ABS(ra_error) < %s"%str(err_pos)
        query +=" AND ABS(dec_error) < %s"%str(err_pos)
        query +=" AND ABS(parallax_error) < %s"%str(err_pos)
        query +=" AND ABS(phot_g_mean_mag) < %s"%str(g_lim)
        #print('\n\t'+query)
        
        data=[[-999]]
        while data[0][0] == -999:
            data = self.get_results(query)
        return(data)
    
# %%    
    def astrometric_query_circle(self, point, radius, err_pos, g_lim):
        """Construct a query of a sky circle centered in point specified when creating this object
            
        Parameters:
        -----------
        radius: Radius of the circle to query
        
        Returns
        --------
        ADQL Query
        data retrieved from the query
        """
        query = "SELECT %s"%', '.join(self.attributes)
        query += " FROM gaiadr2.gaia_source"
        query += " WHERE CONTAINS(POINT('%s', ra, dec), CIRCLE('%s', %s, %s))=1"%(self.coordsys, 
                                 self.coordsys, ', '.join([str(n) for n in point]), str(radius))
        query +=" AND parallax IS NOT NULL"
        query +=" AND parallax >= 0.0"
        query +=" AND ra IS NOT NULL"
        query +=" AND dec IS NOT NULL"
        query +=" AND pmra IS NOT NULL"
        query +=" AND pmdec IS NOT NULL"
        query +=" AND ABS(ra_error) < %s"%str(err_pos)
        query +=" AND ABS(dec_error) < %s"%str(err_pos)
        query +=" AND ABS(parallax_error) < %s"%str(err_pos)
        query +=" AND ABS(phot_g_mean_mag) < %s"%str(g_lim)
        print('\t'+query)                           
        data = self.get_results(query)
        return(data)
        
# %%    
    def get_results(self, query):
        """Runs a job in GAIA archive to retrieve data
        
        Parameters.
        -----------
        query: Query for GAIA database
        
        Returns
        --------
        data: Data retrieved from ADQL query
        """
        data = [[-999]]       
        try:
            job = Gaia.launch_job_async(query)
        except:
            print('Query has failed')
            time.sleep(30)            #Waits for Gaia service to restore before running again!!
        else:
            data = job.get_results()
        return(data)

    
    
# %% ############   DATA EXTRACTION Section   ############
# %%

def extract_data(r, err_pos, g_lim, cluster=None, coord_icrs=None, coord_gal = None):
    '''
    Bruno Femenía Castellá

    Parameters
    ----------
    r : TYPE
        DESCRIPTION.
    err_pos : TYPE
        DESCRIPTION.
    g_lim : TYPE
        DESCRIPTION.
    cluster : TYPE, optional
        DESCRIPTION. The default is None.
    coord_icrs : TYPE, optional
        DESCRIPTION. The default is None.
    coord_gal : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    data      :  data retrived from query to Gaia.
    center    :  ICRS coordinates (ra & dec)
    center_gal: Galactirc coordinates corresponding to center
    '''

    # Define variables needed to make the query 
    data = None
    dias_catalog = DiasCatalog('Cluster', 'RAJ2000', 'DEJ2000', 'l', 'b', 'Class',
                               'Diam', 'Dist', 'pmRA', 'pmDE', 'Nc', 'RV', 'o_RV',
                               l0=coord_gal[0], b0=coord_gal[1], radius=r)
    
    if (coord_icrs != None) or (cluster != None):
        if coord_icrs == None:
            point = dias_catalog.get_ra_dec(cluster)
        else:
            point = coord_icrs
        #Create the GAIA query
        attributes = ['source_id', 'ra', 'ra_error', 'dec', 'dec_error', 'l', 'b',
                      'parallax', 'parallax_error', 'pmra', 'pmdec','phot_g_mean_mag']
        gaia_query = GaiaData(attributes)
        data       = gaia_query.astrometric_query_box(   point, 2*r, err_pos, g_lim)
        #data      = gaia_query.astrometric_query_circle(point,   r, err_pos, g_lim)

    return(data, point)



# %% ############   DATA PREPROCESSing Section   ############
# %%

def preprocessing(data, sample_fctr=1, threshold=9500):

    global sample_factor
    sample_factor = sample_fctr
    
    # 1/ transforms parallax data into distances in parsec
    data_metrics = data
    data_metrics['parallax'] = data_metrics['parallax'].to(u.parsec, equivalencies=u.parallax())
    
    # 2/ Adapt data metrics to a numpy array
    np_data_metrics = np.transpose(np.array([data_metrics['ra'], data_metrics['dec'], data_metrics['parallax'],
                                             data_metrics['pmra'], data_metrics['pmdec']]))
    
    # 3/ Sample data: if a specific sampling is requested or too many points
    n_points= len(np_data_metrics)
    if n_points*sample_factor > threshold:
        sample_factor=threshold/n_points
     
    if(sample_factor != 1):
        np.random.seed(0)
        idx = np.random.choice( np_data_metrics.shape[0], replace= False,
                               size=int(sample_factor*np_data_metrics.shape[0]) )
        np_data_metrics = np_data_metrics[idx,:]
        
    # 4/ Change coordinates from Spherical to Cartesian coordinate system
    ra = Longitude(np_data_metrics[:,0], unit=u.deg)
    ra.wrap_angle = 180 * u.deg
    dec  = Latitude(np_data_metrics[:,1], unit=u.deg)
    dist = Distance(np_data_metrics[:,2], unit=u.parsec)
    sphe_coordinates = SkyCoord(ra, dec, distance = dist, frame='icrs', representation_type='spherical')
    cart_coordinates = sphe_coordinates.cartesian
    
    # 5/ Adapt data to normalize it correctly
    data_sphe_adapted = np.transpose(np.array([sphe_coordinates.ra, sphe_coordinates.dec, sphe_coordinates.distance]))
    data_cart_adapted = np.transpose(np.array([cart_coordinates.x, cart_coordinates.y, cart_coordinates.z]))
    data_pm_adapted = np_data_metrics[:,3:5]   
    data_all_adapted = np.append(data_cart_adapted, data_pm_adapted, axis=1)
    
    return(data_sphe_adapted, data_cart_adapted, data_all_adapted)



# %% ############   DBSCAN Section   ############
# %%

def get_distances_for_ML(X, Y):
    distance = 0
    ra1 = X[0]*u.deg
    ra2 = Y[0]*u.deg
    dec1 = X[1]*u.deg
    dec2 = Y[1]*u.deg
    if(len(X) == 3):
        dist1 = X[2]*u.parsec
        dist2 = Y[2]*u.parsec
        point1 = SkyCoord(ra1, dec1, dist1)
        point2 = SkyCoord(ra2, dec2, dist2)
        distance = point1.separation_3d(point2)
    else:
        point1 = SkyCoord(ra1, dec1)
        point2 = SkyCoord(ra2, dec2)
        distance = point1.separation(point2)
    return(distance.value)


# %% 
def get_distance_matrix(data, scale=False, metric='euclidean'):
    if scale:
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data
        
    if metric != 'euclidean':
        dist_matrix = pairwise_distances(data_scaled, metric=get_distances_for_ML, n_jobs=-1)
    else:
        dist_matrix = pairwise_distances(data_scaled, metric='euclidean', n_jobs=-1)
 
    return(dist_matrix)


# %%
def DBSCAN_eval(data, center, center_gal,r,sample_factor,ftype,dim3, eps_range, 
                min_samples_range, scale=False, metric='euclidean', pm=False):
    
    data_sphe, data_cart, data_all = preprocessing(data, sample_factor)
    data_search = data_sphe[:,:2]
    if ftype == 'cart':
        if dim3:
            data = data_cart
        else:
            data = data_cart[:,:2]
    elif ftype == 'sphe':
        if dim3:
            data = data_sphe
        else:
            data = data_sphe[:,:2]
    elif ftype == 'pm':
        data = data_all
    else:
        print('Specify tpye of dataset: cart, sphe, pm')
        sys.exit()
    
    dist_matrix = get_distance_matrix(data, scale, metric)
    
    if pm:
        data = data[:,:3]
    
    dias_catalog = DiasCatalog('Cluster', 'RAJ2000', 'DEJ2000', 'Class', 'l', 'b',
                               'Diam', 'Dist', 'pmRA', 'pmDE', 'Nc', 'RV', 'o_RV',
                               l0=center_gal[0], b0=center_gal[1], radius=r)
    
    num_cum         = len(dias_catalog.get_clusters(center, r))
    sscores         = {}      
    tmp_d_sscores   = {}                                        # Temp Dict sscores
    tmp_l_sscores   = []                                        # Temp List sscores within tmp_d_sscores
    matches         = []
    cluster_centers = []
    index           = 0
    
    for eps in eps_range:
        
        for min_samples in min_samples_range:            
            db     = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1).fit(dist_matrix)
            labels = db.labels_
            for i in range(len(set(labels))-1):
                cluster        = data_search[np.where(labels == i)]
                cluster_center = cluster.mean(axis=0)
                cluster_centers.append(cluster_center)

            tmp_l_sscores.append( eps            )
            tmp_l_sscores.append( min_samples    )
                
            matches = dias_catalog.get_match(center, r, cluster_centers)
            if(len(matches) > 0 and len(set(labels)) > 1):
                tmp_l_sscores.append(len(matches)/(num_cum + (len(cluster_centers)-len(matches))))
            else:
                tmp_l_sscores.append(0.0)
                
            tmp_l_sscores.append( cluster_centers)
            tmp_d_sscores = {index:tmp_l_sscores}
                
            sscores = {**sscores, **tmp_d_sscores}  # Merging two dicts
            
            index += 1                              # Increasing index for next iteration
            tmp_d_sscores   = {}                    # Resetting        for next iteration
            tmp_l_sscores   = []                    # Resetting        for next iteration
            cluster_centers = []                    # Resetting        for next iteration
            
    return(sscores, dist_matrix, data_search)


# %%    MAIN CODE    ##########################################################
# %%   This to be uncommeted if wishing to run from terminal.

# %%
#if __name__ == "__main__":
#    main()


# %%
# %%
# %% ####################    REDUCER    ####################

def run_ip_file_test(ip_file):
    t0 = time.time()
    print(f'Faking parallel execution of file >>{ip_file}<<')
    for j in range(10):
        dummy = np.random.randn(8192,8192)
        dummy[0]=1
    print(f'\t\tDone with this fake iteration. Seconds: {time.time()-t0}\n\n')

        
    
def run_ip_file(ip_file='ip_file_000001.csv', ftype='sphe'):

    #Parsing filename
    #----------------
    ip_file=ip_file.split(sep='\n')[0]
    ip_file=ip_file.split(sep='\t')[0]
    
    # From ip_file loading I/P parameters into PAndas dataframe: params
    #-----------------------------------------------------------
    params = pd.read_csv('/home/TFM/'+ip_file)
    
    #These lines directly copied from deprectaeed functions: process_line & run_ip_file
    #----------------------------------------------------------------------------------
    ra            = params['ra'][0]
    dec           = params['dec'][0]
    l             = params['l'][0]
    b             = params['b'][0]
    radius        = params['r'][0]
    err_pos       = params['err_pos'][0]
    g_lim         = params['g_lim'][0]
    sample_factor = params['sample'][0]
    scale         = params['norm'][0] == 'X'
    metric        = params['distance'][0]
    dim3          = params['dim3'][0] == 'X'
    eps_min       = params['eps_min'][0]
    eps_max       = params['eps_max'][0]
    eps_num       = params['eps_num'][0]
    min_pts_min   = params['min_pts_min'][0]
    min_pts_max   = params['min_pts_max'][0]
    min_pts_num   = params['min_pts_num'][0]
    eps_range     = np.linspace(eps_min, eps_max, eps_num)
    min_pts_range = np.linspace(min_pts_min, min_pts_max, min_pts_num)
    center_icrs   = [ra, dec]
    center_gal    = [l,  b]

    if sample_factor == None:
        sample_factor = 1


    # print(f'\n\nReading file {ip_file}    #eps: {eps_num}    #Npts: {min_pts_num}') #For TESTING purposes
    
    # Running ETL process: query(Extraction), Loading and Transformation
    #---------------------
    data, center  = extract_data(radius, err_pos, g_lim, coord_icrs=center_icrs, coord_gal=center_gal)
    # print('\t Done with extract_data')

    # Running DBSCAN process: -> Main O/P is param_scores
    #----------------------------------------------------
    param_scores, dist_matrix, data_search = DBSCAN_eval(data, center, center_gal, radius, sample_factor,
                                                         ftype, dim3, eps_range, min_pts_range, scale, metric)
    # print('\t Done with DBSCAN_eval')
    
    # Reporting execution times: This to be commentedo ut when running within MapRed in HAdoop
    #-----------------------------
    # print("\n\n\t\t\t\t Program execution: 9100 iters  ---> %s seconds ---" % (t1 - t0))
    # print("\t\t\t\t Saving results into csv        ---> %s seconds ---" % (t2 - t1))
    # print("\n\t  >>> TOTAL time to process a single I/P file:  %s seconds ---" % (t2- t0))

    # Analyzing DBSCAN results: this to be commented out if using withihn MapRed Hadoop
    #----------------------------------------------------------------------------------
    # DBSCAN_result(param_scores, dist_matrix, data_search, center, radius)


    ###############
    #### IMPORTANT: Uncommnet following lines to execute in MapRed in Hadoop
    ###############
    
    #----------------------------------------------------------------------
    # >> FOR MAPRED in HAdoop: this prints results into stdout as requested
    #    by Hadoop Streaming!!
    #
    #  Printing O/P on stderr to generate results file in HFS
    #  (when executed from within cluster with Hadoop)
    #----------------------------------------------------------------------
    for k, v in param_scores.items():
        print(ip_file, ",", ra, ",", dec, ",", l, ",", b, ",", radius, ",", g_lim,
               ",", sample_factor, ",",  v[0], "," , v[1], "," , v[2], end=', ')  
        if v[2] > 0:
            for coord in v[3]:
                print(coord[0], ',', coord[1], end=', ')
        print()
        
        
#run_ip_file('ip_file_DC_00739.csv')

# %%
# %%
# %%    REDUCER    ############################################################
#
# Reducer goes here!!
#-------------------
for value in sys.stdin:
    run_ip_file(value)
