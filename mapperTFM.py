#!/usr/bin/env python3

#very simple MAPPER function: identify function....
#no key and as a value the IP filename for the reduce routine

#from astroquery.vizier import Vizier  #To check whether this
#line is also triggereing MAPPER to fail: YES, error occurs

import sys
for line in sys.stdin:
    print(line, end='')
    #ip_file= line.split(sep='\n')[0]
    #ip_file= ip_file.split(sep='\t')[0]
    
    #print("{0}\t{1}".format(ip_file,ip_file))
    #print("{0}".format(ip_file))
