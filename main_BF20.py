#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 11:23:54 2019

@author: rafa

Modified by: Bruno Femenía Castellá (bruno.femenia@gmail.com)
"""
import sys
import numpy as np
import pandas as pd
import GaiaData as gd
import DiasCatalog as dc
import datetime

# %%    IMPORT Section     ####################################################

# Third-party dependencies
from astropy import units as u
from astropy.coordinates import Angle, Longitude, Latitude, Distance
from astropy.coordinates import SkyCoord
from astropy.table import Table


# Set up matplotlib and use a nicer set of plot parameters
import matplotlib.pyplot as plt
from matplotlib import cm
from astropy.visualization import astropy_mpl_style
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
    

#Sklearn algorithm
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score


#To suppress warnings
import warnings
warnings.filterwarnings("ignore")


#To time code execution
import time


# %%     DATA EXTRACTION     ##################################################
def extract_data(r, err_pos, g_lim, cluster=None, coordinates=None):
    #Define variables needed to make the query 
    data = None
    dias_catalog = dc.DiasCatalog('Cluster','RAJ2000','DEJ2000','Class','Diam',
                                  'Dist', 'pmRA', 'pmDE', 'Nc', 'RV', 'o_RV',
                                  ra_0=coordinates[0], dec_0=coordinates[1],r_0=r)
    if (coordinates != None) or (cluster != None):
        if coordinates == None:
            point = dias_catalog.get_ra_dec(cluster)
        else:
            point = coordinates
        #Create the GAIA query
        attributes = ['source_id', 'ra', 'ra_error', 'dec', 'dec_error', 'parallax', 'parallax_error', 'pmra', 'pmdec','phot_g_mean_mag']
        gaia_query = gd.GaiaData(attributes)
        data = gaia_query.astrometric_query_circle(point, r, err_pos, g_lim)
    return(data, point)


# %%     DATA PREPROCESSING     ###############################################
def preprocessing(data, sample_factor = None):
    #Preprocessing
    data_metrics = data
    data_metrics['parallax'] = data_metrics['parallax'].to(u.parsec, equivalencies=u.parallax())
    #Adapt data metrics to a numpy array
    np_data_metrics = np.transpose(np.array([data_metrics['ra'], data_metrics['dec'], data_metrics['parallax'],
                                             data_metrics['pmra'], data_metrics['pmdec']]))
    #Sample data
    if(sample_factor != None):
        np.random.seed(0)
        idx = np.random.choice(np_data_metrics.shape[0], size=int(sample_factor*np_data_metrics.shape[0]),
                               replace= False)
        np_data_metrics = np_data_metrics[idx,:]
    #Change coordinates from Spherical to Cartesian coordinate system
    ra = Longitude(np_data_metrics[:,0], unit=u.deg)
    ra.wrap_angle = 180 * u.deg
    dec = Latitude(np_data_metrics[:,1], unit=u.deg)
    dist = Distance(np_data_metrics[:,2], unit=u.parsec)
    sphe_coordinates = SkyCoord(ra, dec, distance = dist, frame='icrs', representation_type='spherical')
    cart_coordinates = sphe_coordinates.cartesian
    #Adapt data to normalize it correctly
    data_sphe_adapted = np.transpose(np.array([sphe_coordinates.ra, sphe_coordinates.dec, sphe_coordinates.distance]))
    data_cart_adapted = np.transpose(np.array([cart_coordinates.x, cart_coordinates.y, cart_coordinates.z]))
    data_pm_adapted = np_data_metrics[:,3:5]   
    data_all_adapted = np.append(data_cart_adapted, data_pm_adapted, axis=1)
    return(data_sphe_adapted, data_cart_adapted, data_all_adapted)

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


# %%     DATA EVALUATON     ###################################################

def generate_ip_dias(l_max=5, r_max=1.5, n_min=3):
    """
    This routine generates the selection of ip files over the Galactic plane
    containing at least 3 clusters in Dias catalog and not beyond r_max degs of
    size. Each of these sky patches witll be analyzed to derived eps=eps(l,b) 
    and min_pts=min_pets(l,b) which will guide the search for clusters in a 
    subsequent exploration of Gaia data.
    
    Based on these files we evaluate the (eps,Nmin) to be used over different regions of whole
    Galactic plane to find new candidates with DBSCAN.

    Returns
    -------
    None.

    """
    
    full_dc=dc.DiasCatalog('Cluster','RAJ2000','DEJ2000','Class','Diam',entire=True,galactic=True)
    
    gal_disk=full_dc.table[ (abs(full_dc.table['b']) < 20) & (full_dc.table['l'] < l_max) ]
   
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
             'g_lim':[18.5],
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
        
                                
# %%
def DBSCAN_result(param_scores, dist_matrix, data, center, radius):
    #dias_catalog = dc.DiasCatalog()
    dias_catalog = dc.DiasCatalog('Cluster','RAJ2000','DEJ2000','Class','Diam',
                                  'Dist', 'pmRA', 'pmDE', 'Nc', 'RV', 'o_RV',
                                  ra_0=center[0], dec_0=center[1],r_0=radius)
    cluster_centers = []
    sorted_param_scores = param_scores.sort_values(by=['local_score'], ascending=False)
    opt_epsilon = sorted_param_scores.iloc[0,0]
    opt_min_pts = sorted_param_scores.iloc[0,1]
    db = DBSCAN(eps=opt_epsilon, min_samples=opt_min_pts, metric='precomputed', n_jobs=-1).fit(dist_matrix)
    labels = db.labels_
    for i in range(len(set(labels))-1):
        cluster = data[np.where(labels == i)]
        cluster_center = cluster.mean(axis=0)
        cluster_centers.append(cluster_center)
    matches = dias_catalog.get_match(center, radius, cluster_centers)
    for m in matches:
        print('Cluster found: %s'%(m[0]))
    plot_clusters(data, labels)
    plot_score(param_scores)


# %%
def DBSCAN_eval(data, center, r, sample_factor, ftype, dim3, eps_range, min_samples_range, scale=False, metric = 'euclidean', pm=False):
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
    
    #dias_catalog = dc.DiasCatalog()
    dias_catalog = dc.DiasCatalog('Cluster','RAJ2000','DEJ2000','Class','Diam',
                                  'Dist', 'pmRA', 'pmDE', 'Nc', 'RV', 'o_RV',
                                  ra_0=center[0], dec_0=center[1],r_0=r)    
    num_cum = len(dias_catalog.get_clusters(center, r))
    tmp_sscores = []
    matches = []
    sscores = pd.DataFrame(columns=['epsilon', 'minpts', 'local_score'])
    cluster_centers = []
    
    # idx_eps=0
    
    for eps in eps_range:
        tA= time.time()
        # idx_pts=0
        print(f'\nCase eps: {eps}\n'+9*'-')
        
        for min_samples in min_samples_range:            
            # tB=time.time()
            db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1).fit(dist_matrix)
            # tC=time.time()
            # print(f'\n\tSeconds to execute DBSCAN:{tC-tB}')
            
            labels = db.labels_
            for i in range(len(set(labels))-1):
                cluster = data_search[np.where(labels == i)]
                cluster_center = cluster.mean(axis=0)
                cluster_centers.append(cluster_center)
                
            # tD=time.time()
            matches = dias_catalog.get_match(center, r, cluster_centers)
            # tE=time.time()
            # print(f'\tSeconds to execute dias_catalog.get_match:{tE-tD}')
            
            tmp_sscores.append(eps)
            tmp_sscores.append(min_samples)
            if(len(matches) > 0 and len(set(labels)) > 1):
                tmp_sscores.append(len(matches)/(num_cum + (len(cluster_centers)-len(matches))))
            else:
                tmp_sscores.append(0.0)
            sscores.loc[len(sscores)] = tmp_sscores
            tmp_sscores = []
            cluster_centers = []
        
            # print(f'\tSeconds to execute (eps,Npts)=({eps},{min_samples}): {time.time()-tB}')
            # idx_pts +=1
            # if idx_pts==2:
                # break
      
        print(f'\t>>  Seconds to execute eps={eps}: {time.time()-tA}')
        # idx_eps += 1
        # if (idx_eps == 2):
        #     break
        
    return(sscores, dist_matrix, data_search)



# %% DATA PLOTTING      #######################################################
def plot_clusters(X, labels):
    # Black removed and is used for noise instead.
    size = 6.0
    f = plt.figure(figsize=(20,20))
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 0.5, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
            size = 3.5

        class_member_mask = (labels == k)

        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=size)
        size=6.0

    plt.title('Estimated number of clusters: %5d' % int(len(set(labels))-1))
    plt.xlabel("Right Ascension (deg)")
    plt.ylabel("Declination (deg)")
    
    dummy  = datetime.datetime.now()
    op_name= f'plt_clstr_exec_{dummy.year}-{dummy.month}-{dummy.day} {dummy.hour}-{dummy.minute}-{dummy.second}.png'
    f.savefig(op_name)

    
# %%    
def plot_score(param_scores):
    np_param_scores = param_scores.values
    eps = np.sort(np.array(list(set(param_scores['epsilon']))))
    Nmin = np.sort(np.array(list(set(param_scores['minpts']))))
    Z = np.empty((len(Nmin), len(eps)))
    fig, ax = plt.subplots(constrained_layout = True)
    X, Y = np.meshgrid(eps, Nmin)
    for i, n in enumerate(Nmin):
        for j, e in enumerate(eps):
            Z[i,j] = np_param_scores[np.where((param_scores['epsilon'] == e) & (param_scores['minpts'] == n)),2]
    extend = "neither"
    cmap = plt.cm.get_cmap('hot')
    CS = ax.contourf(X,Y,Z, cmap=cmap, extend=extend)
    fig.colorbar(CS)
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('Nmin')
    ax.set_title('DBSCAN matching M')
    
    dummy  = datetime.datetime.now()
    op_name= f'plt_score_exec_{dummy.year}-{dummy.month}-{dummy.day}.png'
    fig.savefig(op_name)
    #fig.savefig('plot_score_execution_%s.png'%(datetime.date.today()))

# %%
def check_errors(f):
    def check(*args, **kwargs):
        try:
            f(*args, **kwargs)
        except:
            print("An error has ocurred")
    return(check)


# %%
@check_errors
def plot_bar(*args):
    if args[1] == 'eps':
        df = args[0][['epsilon','local_score']]
        df['bucket'] = pd.cut(df.epsilon, args[2])
        title = 'Máximo M en función de epsilon'
        x_label = 'epsilon'
    else:
        df = args[0][['minpts','local_score']]
        df['bucket'] = pd.cut(df.minpts, args[2])
        title = 'Máximo M en función de Nmin'
        x_label = 'Nmin'
    newdf = df[['bucket','local_score']].groupby('bucket').max()   
    ax = newdf.plot(kind='bar', title=title, colormap = 'jet', fontsize=7, legend=False)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Max(M)")
    ax.grid()    


#%%    MAIN     ###############################################################

def process_line(ftype, params):
    ra = params['ra']
    dec = params['dec']
    radius = params['r']
    err_pos = params['err_pos']
    g_lim = params['g_lim']
    sample_factor = params['sample']
    scale = params['norm'] == 'X'
    metric = params['distance']
    dim3 = params['dim3'] == 'X'
    eps_min = params['eps_min']
    eps_max = params['eps_max']
    eps_num = params['eps_num']
    min_pts_min = params['min_pts_min']
    min_pts_max = params['min_pts_max']
    min_pts_num = params['min_pts_num']
    eps_range = np.linspace(eps_min, eps_max, eps_num)
    min_pts_range = np.linspace(min_pts_min, min_pts_max, min_pts_num)
    
    center = [ra, dec]
    data, center = extract_data(radius, err_pos, g_lim, coordinates = center)
    param_scores, dist_matrix, data_search = DBSCAN_eval(data, center, radius, sample_factor, ftype, dim3, eps_range, min_pts_range, scale, metric)

    return(param_scores, dist_matrix, data_search, center, radius)
    
    
# %%
def main():
    if len(sys.argv) == 3:
        file = sys.argv[1]
        ftype = sys.argv[2]
        try:
            params = pd.read_csv(file)
        except:
            print('The file does not exist')
            sys.exit(0)
        for i in range(len(params)):
            param_scores, dist_matrix, data_search, center, radius = process_line(ftype, params.iloc[i,:])
            param_scores.to_csv('execution%s_%s.csv'%(str(i), datetime.date.today()))
            DBSCAN_result(param_scores, dist_matrix, data_search, center, radius)

            
# %%    
#    This is just to launch the mainn code from a session with Spyder IDE
        
def main_spyder(file='ip_file_0001.csv', ftype='sphe'):
    
    file=file.split(sep='\n')[0]
    print('\n\n\nStarting main_spyder!!\n'+22*'='+'\n\n')
    try:
        print(f'\tReading file {file}\n')
        params = pd.read_csv(file)
    except:
        print('The file does not exist')
        return
            

    for i in range(len(params)):
        t0 = time.time()
        print('\nStarting Iter# {0:1.0f}'.format(i+1))
        print(16*'=')
        print("  Sending query to GAIA\n ", 21*'-')
        
        param_scores, dist_matrix, data_search, center, radius = process_line(ftype, params.iloc[i,:])
        t1= time.time()
        print("\n\n     Execution of process_line    --- %s seconds ---" % (t1 - t0))
        
        param_scores.to_csv('execution%s_%s.csv'%(str(i), datetime.date.today()))
        t2= time.time()
        print("     Saving results into csv      --- %s seconds ---" % (t2- t1))
        
        DBSCAN_result(param_scores, dist_matrix, data_search, center, radius)
        t3= time.time()
        print("     Execution of DBSCAN_result   --- %s seconds ---" % (t3-t2))
        
        print("  >> Execution of this iterarion  --- %s seconds ---" % (t3- t2))


#main_spyder()


# %%
if __name__ == "__main__":
    main()