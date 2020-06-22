#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 10:59:51 2018

@author: rafa

This class fetches data from Gaia depending on certain paramters:
"""

from astroquery.gaia import Gaia

class GaiaData(object):
 
# %%    
    def __init__(self, attributes, coordsys = 'ICRS'):
        """Constructo of an object representing an ADQL query
            
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
        print(query)                           
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
        else:
            data = job.get_results()
        return(data)
        
        