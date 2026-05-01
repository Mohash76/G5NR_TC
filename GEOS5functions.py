import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#import arrow   #https://arrow.readthedocs.io/en/latest/
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp2d
import matplotlib.path as mpath
import xarray as xr
#import pyke
#import iris
#import iris.plot as iplt
#import iris.util
#import iris.quickplot as qplt
#import iris.analysis.cartography
#from iris.analysis.cartography import rotate_pole, rotate_winds
#import iris.coord_systems
from TCcatalog_locations import *
from TCstormvectors import *
import datetime
import dask
from dask import delayed
from concurrent.futures import ProcessPoolExecutor

# Functions listed in this order:
    # get_hurricane
    # UpperTroposphericHumidity
    # getDistLatLong
    # gettheta
    # bincylinder
    # bincylinderstream
    # removeadvection
    # xydist
    # haversineSY
    # haversine
    # radialwinds
    # getallstorms
    # whichstorms 
    # oceanlimits
    
    #DEFUNCT:
    # radialwindsold


allstormsvec = [atl05vec,atl06vec,epc05vec,epc06vec,npc05vec,npc06vec,nio05vec,nio06vec,sio56vec,sio67vec,aus56vec,aus67vec]

#################################################################################################
def findboxhurricanedays(lat1,lat2,lon1,lon2,timevec,substormsvec,degbuffer):
    
    stormlist = getallstorms()
    stormlocind = 0
    all_lons = []
    all_lats = []
    all_pres = []
    all_time = []
    
    count = 0
    for tt in timevec:
        stormlocind = 0
        templons = []
        templats = []
        temppres = []
        temptime = []
        for storm in substormsvec:
            plotlon = np.nan
            plotlat = np.nan
            extrastormname = stormlist[1][storm]
            loadthisfile = '/home/groups/oneillm/groupshare/forHoward/' + extrastormname[0:9] + '.npz'
            extranpzfile = np.load(loadthisfile)
            extratctime  = extranpzfile['time']
            extraminplon = extranpzfile['minplon']
            extraminplat = extranpzfile['minplat']
            extraminpres = extranpzfile['minpres']
            extratimestringfirst = np.datetime64(extratctime[0])
            extratimestringlast  = np.datetime64(extratctime[-1])
            if extratimestringfirst <= tt:
                if extratimestringlast >= tt:
                    for extraind in range(len(extratctime)):
                        extratimestring = np.datetime64(extratctime[extraind])
                        #print('now about to compare plot time',tt,' and tctrack time',extratimestring)
                        if extratimestring < tt + np.timedelta64(5,'m'):
                            if extratimestring > tt - np.timedelta64(5,'m'):
                                #print('we have a time match!')
                                if extraminplon[extraind] >= (lon1-degbuffer) and extraminplon[extraind] <= (lon2+degbuffer):
                                    if extraminplat[extraind] >= (lat1-degbuffer) and extraminplat[extraind] <= (lat2+degbuffer):
                                        templons.append(extraminplon[extraind])
                                        templats.append(extraminplat[extraind])
                                        temppres.append(extraminpres[extraind])
                                        temptime.append(tt)
                                        stormlocind = stormlocind + 1
            del loadthisfile, extranpzfile, extratctime,extraminplon,extraminplat,extraminpres
        if stormlocind == 0:
            templons.append(np.nan)
            templats.append(np.nan)
            temppres.append(np.nan)
            temptime.append(np.nan)
        all_lons.append(templons)
        all_lats.append(templats)
        all_pres.append(temppres)
        all_time.append(temptime)
        
    return all_lons, all_lats, all_pres, all_time
    
    

#################################################################################################
def get_hurricane():
    u = np.array([  [2.444,7.553],
                    [0.513,7.046],
                    [-1.243,5.433],
                    [-2.353,2.975],
                    [-2.578,0.092],
                    [-2.075,-1.795],
                    [-0.336,-2.870],
                    [2.609,-2.016]  ])
    u[:,0] -= 0.098
    codes = [1] + [2]*(len(u)-2) + [2] 
    u = np.append(u, -u[::-1], axis=0)
    codes += codes

    return mpath.Path(3*u, codes, closed=False)

hurricane = get_hurricane()

#################################################################################################
def LayerAverage(var,delp,ind1,ind2):
    var = np.mean(var[ind1:ind2,:,:]*delp[ind1:ind2,:,:],axis=0)/np.mean(delp[ind1:ind2,:,:])
    return var;

#################################################################################################
# Some useful functions for computing distances and angle w.r.t. center
def getDistLatLong(lat1,lon1,lat2,lon2): #written by Ipshita Dey, 2020
    n1 = np.size(lat1); n2 = np.size(lat2)
    if ((n1>1) & (n2>1)):
        d  = np.nan
    elif ((n1>1) & (n2==1)):
        lat1 = np.array(lat1); lon1 = np.array(lon1)
    else:
        lat2 = np.array(lat2); lon2 = np.array(lon2)
    R = 6371; #Radius of the earth in km
    dLat = np.deg2rad(lat2-lat1);   
    dLon = np.deg2rad(lon2-lon1); 
    a = (np.sin(dLat/2))**2 + np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * (np.sin(dLon/2))**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)); 
    d = R * c; # Distance in km

    return d;

#################################################################################################
def gettheta(dx,dy): #written by Ipshita Dey, 2020
    if (np.size(dx) == 1):
        dx = float(dx); dy = float(dy)
        if ((dx == 0) & (dy>=0)): theta = 90.0
        if ((dx == 0) & (dy<0)): theta = 270.0
        if (dx<0): theta = 180.0 + np.rad2deg(np.arctan(dy/dx))
        if ((dx > 0) & (dy >= 0)): theta = np.rad2deg(np.arctan(dy/dx))
        if ((dx > 0) & (dy < 0)): theta = 360.0 + np.rad2deg(np.arctan(dy/dx))
    else:
        dx = dx.astype(float); dy = dy.astype(float)
        theta = np.zeros_like(dx)
        ind = np.where((dx == 0) & (dy>=0))
        theta[ind] = 90
        ind = np.where((dx == 0) & (dy<0))
        theta[ind] = 270   
        ind = np.where(dx < 0)
        theta[ind] = 180 + np.rad2deg(np.arctan(dy[ind]/dx[ind]))
        ind = np.where((dx > 0) & (dy >= 0))
        theta[ind] = np.rad2deg(np.arctan(dy[ind]/dx[ind]))
        ind = np.where((dx > 0) & (dy < 0))
        theta[ind] = 360 + np.rad2deg(np.arctan(dy[ind]/dx[ind]))
    return theta;

#################################################################################################
def bincylinder(var,rotated_lats,r,area): #written by Shanni You, 2020
    
    dshape = (var.shape[1]-1,len(r))

    azlvar = np.zeros(dshape)

    all_area = np.zeros(var.shape)
    for i in range(var.shape[1]):
        all_area[0,i,:,:]=area[0,:,:]

    for i in range(r.shape[0]-1):
        ring = np.where((np.array(rotated_lats) < r[i]) & 
                                  (np.array(rotated_lats) >= r[i+1]), 1, 0)
        print(i)
        num = np.sum(np.sum(ring*all_area[0,1,:,:],axis = 1),axis = 0)
        azlvar[:,i] = np.sum(np.sum(ring*var[0,1:,:,:]*all_area[0,1:,:,:],axis = 2),axis = 1)/num
    
    return azlvar
#################################################################################################

def bincylinderstream(radwind,tanwind,wwind,rho,h,rotated_lats,r,area): #written by Shanni You, 2020
    
    dshape = (h.shape[1]-1,len(r))

    azl_ur = np.zeros(dshape)
    azl_ut = np.zeros(dshape)
    azl_w = np.zeros(dshape)
    azl_rho = np.zeros(dshape)
    azl_h = np.zeros(dshape)
    count = 0
    all_area = np.zeros(h.shape)
    for i in range(h.shape[1]):
        all_area[0,i,:,:]=area[0,:,:]

    for i in range(r.shape[0]-1):
        ring = np.where((np.array(rotated_lats) < r[i]) & 
                                  (np.array(rotated_lats) >= r[i+1]), 1, 0)
        print(str(i)+', np.where just created ring variable')

        num = np.nansum(np.nansum(ring*all_area[0,1,:,:],axis = 1),axis = 0)
        azl_ur[:,i] = np.nansum(np.nansum(ring*radwind[0,1:,:,:]*all_area[0,1:,:,:],axis = 2),axis = 1)/num
        print('azl_ur finished')
        azl_ut[:,i] = np.nansum(np.nansum(ring*tanwind[0,1:,:,:]*all_area[0,1:,:,:],axis = 2),axis = 1)/num
        print('azl_ut finished')
        azl_w[:,i] = np.sum(np.sum(ring*wwind[0,1:,:,:]*all_area[0,1:,:,:],axis = 2),axis = 1)/num
        print('azl_w finished')
        azl_rho[:,i] = np.sum(np.sum(ring*rho[0,1:,:,:]*all_area[0,1:,:,:],axis = 2),axis = 1)/num
        print('azl_rho finished')
        azl_h[:,i] = np.sum(np.sum(ring*h[0,1:,:,:]*all_area[0,1:,:,:],axis = 2),axis = 1)/num
        print('azl_h finished')
        del ring
    
    return azl_ur, azl_ut, azl_w, azl_rho, azl_h

#################################################################################################

def removeadvection(u,v,minplat,minplon,dthr,tstep): #written by Shanni You, 2020
    # using haversine function to solve for the x, y in flat plane:
    # define the center point as the original center point:
    xcyl_mid,ycyl_mid = xydist(minplat[2:],minplon[2:],minplat[:-2],minplon[:-2])
    xcyl_first,ycyl_first = xydist(minplat[1],minplon[1],minplat[0],minplon[0])
    xcyl_last,ycyl_last = xydist(minplat[-1],minplon[-1],minplat[-2],minplon[-2])
    
    # computing for the mean_u,v:
    # veterization version:
    mean_u = np.zeros(minplon[:].shape)
    mean_v = np.zeros(len(minplon[0:]))
    dt = 0.5*60*60 # s-1
    
    mean_u[1:-1] = (0.5/dt) * xcyl_mid
    mean_v[1:-1] = (0.5/dt) * ycyl_mid
    mean_u[0] = (1.0/dt) * xcyl_first
    mean_v[0] = (1.0/dt) * ycyl_first
    mean_u[-1] = (1.0/dt) * xcyl_last
    mean_v[-1] = (1.0/dt) * ycyl_last
    
    u2 = u-mean_u[tstep]
    v2 = v-mean_v[tstep]
    print(mean_u.shape)
    
    return u2, v2
    
#################################################################################################
    
def streamfunction(radwind,wwind,rhofield,rfield,drfield,dzfield):
    Psi_r = np.nancumsum(rfield*rhofield*wwind*drfield,axis = 1)
    Psi_z = np.nancumsum(rfield*rhofield*radwind*dzfield,axis = 0)
    streamfield = 0.5 * (Psi_r + Psi_z)
    
    return Psi_r,Psi_z,streamfield

#################################################################################################

#################################################################################################

# xydist contributed by Shanni You, 25 Nov 2020:
def xydist(lat2,lon2,lat1,lon1):
    midlat = lat1
    midlon = lon2
    latdist = haversineSY(lat1,lon1,midlat,midlon)
    londist = haversineSY(midlat,midlon,lat2,lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    #print(latdist.shape,'shape')
    latdist = np.where(dlat>=0,latdist,-1*latdist)
    #print(dlat,dlon)
    #print(latdist)
    londist = np.where(dlon>=0,londist,-1*londist)
    return londist,latdist

#################################################################################################

# haversineSY contributed by Shanni You, 25 Nov 2020:
def haversineSY(latmat,lonmat,minlat,minlon): #http://www.movable-type.co.uk/scripts/latlong.html
    R = 6371000           # Earth's radius in m
    #kmperlat = 0.9999*R*2*np.pi/360
    
    #PYTHAGORAS' THEOREM
    #From site above: "If performance is an issue and accuracy is less important, for small distances Pythagoras' theorem can be used on an equirectangular projection: (good for when the distance is less than 20 km; https://cs.nyu.edu/visual/home/proj/tiger/gisfaq.html
    #x = delta(lon) * lat_m
    # = delta(lat)
    #d = R*np.sqrt(x**2 + y**2)
    
    #POLAR COORDINATE FLAT-EARTH FORMULA
    #"The Pythagorean flat-Earth approximation assumes that meridians are parallel, that the parallels of latitude are negligibly different from great circles, and that great circles are negligibly different from straight lines. Close to the poles, the parallels of latitude are not only shorter than great circles, but indispensably curved. Taking this into account leads to the use of polar coordinates and the planar law of cosines for computing short distances near the poles" -- https://cs.nyu.edu/visual/home/proj/tiger/gisfaq.html
    #a = np.pi/2 - lat1
    #b = np.pi/2 - lat2
    #c = np.sqrt(a**2 + b**2 -2*a*b*np.cos((lon2-lon1)))
    #d = R * c
              
    #HAVERSINE METHOD
    #The Haversine method is the best. Haversine Formula (from R.W. Sinnott, "Virtues of the Haversine", Sky and Telescope, vol. 68, no. 2, 1984, p. 159): 
    # phi is latitude, lambda is longitude
    
    # a is the square of half the chord length between the points
    a = (np.sin((latmat-minlat)*np.pi/180/2))**2 + np.cos(minlat*np.pi/180)*np.cos(latmat*np.pi/180)*(np.sin((lonmat-minlon)*np.pi/180/2))**2

    # c is the angular distance in radians
    c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
    d = R*c
    
    #solve for local x,y in flat plane with parallel latitudes
    #ycyl = kmperlat*(latmat-minlat)
    #xcyl = np.sqrt(d**2 - ycyl**2)
    #xcyl[:,0:len(lon)//2] = -1.*xcyl[:,0:len(lon)//2]
    
    #When the two points are antipodal (on opposite sides of the Earth), the Haversine Formula is ill-conditioned (see the discussion below the Law of Cosines for Spherical Trigonometry), but the error, perhaps as large as 2 km (1 mi), is in the context of a distance near 20,000 km (12,000 mi).
    
    return d

#################################################################################################

def haversine(lat,latmat,minlat,lon,lonmat,minlon): #http://www.movable-type.co.uk/scripts/latlong.html
    R = 6371000           # Earth's radius in m
    kmperlat = 0.9999*R*2*np.pi/360
    
    #PYTHAGORAS' THEOREM
    #From site above: "If performance is an issue and accuracy is less important, for small distances Pythagoras' theorem can be used on an equirectangular projection: (good for when the distance is less than 20 km; https://cs.nyu.edu/visual/home/proj/tiger/gisfaq.html
    #x = delta(lon) * lat_m
    #y = delta(lat)
    #d = R*np.sqrt(x**2 + y**2)
    
    #POLAR COORDINATE FLAT-EARTH FORMULA
    #"The Pythagorean flat-Earth approximation assumes that meridians are parallel, that the parallels of latitude are negligibly different from great circles, and that great circles are negligibly different from straight lines. Close to the poles, the parallels of latitude are not only shorter than great circles, but indispensably curved. Taking this into account leads to the use of polar coordinates and the planar law of cosines for computing short distances near the poles" -- https://cs.nyu.edu/visual/home/proj/tiger/gisfaq.html
    #a = np.pi/2 - lat1
    #b = np.pi/2 - lat2
    #c = np.sqrt(a**2 + b**2 -2*a*b*np.cos((lon2-lon1)))
    #d = R * c
              
    #HAVERSINE METHOD
    #The Haversine method is the best. Haversine Formula (from R.W. Sinnott, "Virtues of the Haversine", Sky and Telescope, vol. 68, no. 2, 1984, p. 159): 
    # phi is latitude, lambda is longitude
    
    # a is the square of half the chord length between the points
    a = (np.sin((latmat-minlat)*np.pi/180/2))**2 + np.cos(minlat*np.pi/180)*np.cos(latmat*np.pi/180)*(np.sin((lonmat-minlon)*np.pi/180/2))**2

    # c is the angular distance in radians
    c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
    d = R*c
    
    #solve for local x,y in flat plane with parallel latitudes
    ycyl = kmperlat*(latmat-minlat)    
    xcyl = np.sqrt(d**2 - ycyl**2)
    xcyl[:,0:len(lon)//2] = -1.*xcyl[:,0:len(lon)//2]
    
    #When the two points are antipodal (on opposite sides of the Earth), the Haversine Formula is ill-conditioned (see the discussion below the Law of Cosines for Spherical Trigonometry), but the error, perhaps as large as 2 km (1 mi), is in the context of a distance near 20,000 km (12,000 mi).
    return d, xcyl, ycyl

#################################################################################################

#def radialwinds(u,v,lon,lat,centerlon,centerlat):
    vcube = v.to_iris()
    ucube = u.to_iris()

    ulat = ucube.coord("latitude")
    ulon = ucube.coord("longitude")

    ulat.standard_name = "latitude"
    ulon.standard_name = "longitude"
    # ucube
    
    ucube.remove_coord("latitude")
    ucube.add_dim_coord(ulat, 2)
    ucube.remove_coord("longitude")
    ucube.add_dim_coord(ulon, 3)
    # vcube
    vcube.remove_coord("latitude")
    vcube.add_dim_coord(ulat, 2)
    vcube.remove_coord("longitude")
    vcube.add_dim_coord(ulon, 3)

    #https://scitools.org.uk/iris/docs/latest/iris/iris/coord_systems.html?highlight=rotated#iris.coord_systems.RotatedGeogCS

    # https://scitools.org.uk/iris/docs/latest/iris/iris/analysis/cartography.html#iris.analysis.cartography.rotate_pole
    # https://github.com/SciTools/iris/blob/master/lib/iris/analysis/cartography.py
    # rotated_lons, rotated_lats = rotate_pole(lons, lats, pole_lon, pole_lat)

    #     Returns:
    #         An array of rotated-pole longitudes and an array of rotated-pole
    #         latitudes.
    lonmat,latmat = np.meshgrid(lon,lat)
    rotated_lons, rotated_lats = rotate_pole(lonmat, latmat, centerlon, centerlat)
    cs = iris.coord_systems.RotatedGeogCS(centerlat,centerlon)
    
    tanwindtmp,radwindtmp = rotate_winds(ucube,vcube,cs)
    radwindtmp = -1*radwindtmp
    
    tanwind = xr.DataArray.from_iris(tanwindtmp)
    radwind = xr.DataArray.from_iris(radwindtmp)
    del u,v,ucube,vcube,tanwindtmp,radwindtmp
    
    return tanwind, radwind, rotated_lons, rotated_lats

def radialwinds(u, v, lon, lat, centerlon, centerlat):
    vcube = v.to_iris()
    ucube = u.to_iris()

    # Update latitude and longitude coordinates
    for cube in [ucube, vcube]:
        lat_coord = cube.coord('latitude')
        lon_coord = cube.coord('longitude')
        
        lat_coord.standard_name = "latitude"
        lon_coord.standard_name = "longitude"
        
        # Ensure the coordinates are dimensional
        if lat_coord not in cube.dim_coords:
            cube.remove_coord('latitude')
            cube.add_dim_coord(lat_coord, cube.coord_dims(lat_coord)[0])
        
        if lon_coord not in cube.dim_coords:
            cube.remove_coord('longitude')
            cube.add_dim_coord(lon_coord, cube.coord_dims(lon_coord)[0])

    # Rest of the function remains the same
    lonmat, latmat = np.meshgrid(lon, lat)
    rotated_lons, rotated_lats = rotate_pole(lonmat, latmat, centerlon, centerlat)
    cs = iris.coord_systems.RotatedGeogCS(centerlat, centerlon)
    
    tanwindtmp, radwindtmp = rotate_winds(ucube, vcube, cs)
    radwindtmp = -1 * radwindtmp
    
    tanwind = xr.DataArray.from_iris(tanwindtmp)
    radwind = xr.DataArray.from_iris(radwindtmp)
    
    return tanwind, radwind, rotated_lons, rotated_lats






#################################################################################################

def getallstorms():
    
    stormlist = [[[] for i in range(171)] for i in range(9)]
    tccount = 0
    
    for vecind in range(len(allstormsvec)):
        basinvec = allstormsvec[vecind]
        for vecloop in range(len(basinvec)):
            storm = basinvec[vecloop]
            hr1 = storm[0:2]
            mn1 = storm[2:4]
            dy1 = storm[9:11]
            mo1 = storm[12:15]
            yr1 = storm[16:20]
            hr2 = storm[21:23]
            mn2 = storm[23:25]
            dy2 = storm[30:32]
            mo2 = storm[33:36]
            yr2 = storm[37:41]
            sec = '00'
            name= storm[42:52]
            
            lat1str= storm[55:58]
            #print(lat1str)
            lat1 = int(lat1str[0:2])
            if lat1str[2]=='S':
                lat1=-lat1
            #print(lat1)  
            
            lat2str= storm[60:63]
            #print(lat2str)
            lat2 = int(lat2str[0:2])
            if lat2str[2]=='S':
                lat2=-lat2
            #print(lat2)    
            
            lon1str= storm[67:71]
            #print(lon1str)
            lon1 = int(lon1str[0:3])
            if lon1str[3]=='W':
                lon1=-lon1
            #print(lon1)
            
            lon2str= storm[73:77]
            #print(lon2str)
            lon2 = int(lon2str[0:3])
            if lon2str[3]=='W':
                lon2=-lon2
            #print(lon2)
            #print(name)
            
            startstring = str(yr1+'-'+mo1+'-'+dy1+' '+hr1+':'+mn1+':'+sec)
            endstring   = str(yr2+'-'+mo2+'-'+dy2+' '+hr2+':'+mn2+':'+sec)

            dtstart = datetime.datetime.strptime(startstring, '%Y-%b-%d %H:%M:%S')
            dtend   = datetime.datetime.strptime(endstring, '%Y-%b-%d %H:%M:%S')
            
            stormlist[0][tccount] = tccount
            stormlist[1][tccount] = name
            stormlist[2][tccount] = name[0:3] # basin identifier
            stormlist[3][tccount] = dtstart
            stormlist[4][tccount] = dtend
            stormlist[5][tccount] = lat1
            stormlist[6][tccount] = lat2
            stormlist[7][tccount] = lon1
            stormlist[8][tccount] = lon2
            
            tccount = tccount + 1

    return stormlist

#################################################################################################

def whichstorms(time):
    
    currentstorms = []
    stormlist = getallstorms()
    
    for id in range(171):
        dtstart = stormlist[3][id]
        dtend   = stormlist[4][id]
        
        if time >= dtstart:
            if time <= dtend:
                currentstorms.append(id)
    
    return currentstorms
    
  
    
    '''  
    
    for vecind in range(len(allstormsvec)):
        basinvec = allstormsvec[vecind]
        for vecloop in range(len(basinvec)):
            storm = basinvec[vecloop]
            hr1 = storm[0:2]
            mn1 = storm[2:4]
            dy1 = storm[9:11]
            mo1 = storm[12:15]
            yr1 = storm[16:20]
            hr2 = storm[21:23]
            mn2 = storm[23:25]
            dy2 = storm[30:32]
            mo2 = storm[33:36]
            yr2 = storm[37:41]
            sec = '00'
            name= storm[42:52]
            loc = storm[52:77]
            
            startstring = str(yr1+'-'+mo1+'-'+dy1+' '+hr1+':'+mn1+':'+sec)
            endstring   = str(yr2+'-'+mo2+'-'+dy2+' '+hr2+':'+mn2+':'+sec)

            dtstart = datetime.datetime.strptime(startstring, '%Y-%b-%d %H:%M:%S')
            dtend   = datetime.datetime.strptime(endstring, '%Y-%b-%d %H:%M:%S')
  
            if time >= dtstart:
                if time <= dtend:
                    currentstorms.append(startstring + ' to ' + endstring + ' storm ' + name + loc)

    return currentstorms
    '''

#################################################################################################

def oceanlimits(oceanstr):
    
    #Atlantic
    if oceanstr == 'atl':
        latbasinN = 50.
        latbasinS = 10.
        lonbasinW = -100.
        lonbasinE = -15.
        
    #East Pacific
    if oceanstr == 'epc':
        latbasinN = 27.
        latbasinS = 8.
        lonbasinW = -150.
        lonbasinE = -85.
        
    #West North Pacific
    if oceanstr == 'npc':
        latbasinN = 45.
        latbasinS = 5.
        lonbasinW = 105.
        lonbasinE = 179.5 #but one storm exceeds this, starts over the line
        
    #North Indian Ocean
    if oceanstr == 'nio':
        latbasinN = 23.
        latbasinS = 2.
        lonbasinW = 70.
        lonbasinE = 105.
        
    #South Indian Ocean
    if oceanstr == 'sio':
        latbasinN = -2.
        latbasinS = -40.
        lonbasinW = 35.
        lonbasinE = 100.
        
    #Australian region
    if oceanstr == 'aus':
        latbasinN = -8.
        latbasinS = -35.
        lonbasinW = 98.
        lonbasinE = 179.5
        
    return latbasinN,latbasinS,lonbasinW,lonbasinE



####################################
######## DEFUNCT CODE ##############

def radialwindsold(degnum,rinner,rlim,dr,lev,zcoord,lat,latmat,minlat,lon,lonmat,minlon,u,v):
    
     #get distances from center of storm
    d,xcyl,ycyl = haversine(lat,latmat,minlat,lon,lonmat,minlon)
    cyltheta    = np.mod(np.arctan(ycyl,xcyl),2*np.pi)
    
    # Create angle vector
    theta0 = np.linspace(0,2*np.pi,degnum,endpoint=False)
    rad0 = np.linspace(0,rlim,rlim/dr)
    radgrid,thetagrid = np.meshgrid(rad0,theta0)
    
    # Get x,y points around circle
    polarx = radgrid*np.cos(thetagrid) 
    polary = radgrid*np.sin(thetagrid)

    uwind = np.squeeze(u)
    vwind = np.squeeze(v)

    uinterp = np.zeros((len(lev),len(theta0),len(rad0)))
    vinterp = np.zeros((len(lev),len(theta0),len(rad0)))
    #print(lat-minlat.values)
    #print(lon-minlon.values)

    # Loop on vertical levels
    for k in np.arange(len(lev)):
        
        uinterpfunction = interp2d(d,cyltheta,uwind[k,:,:])
        vinterpfunction = interp2d(d,cyltheta,vwind[k,:,:])
        uinterp[k,:,:] = uinterpfunction(rad0,theta0)
        vinterp[k,:,:] = vinterpfunction(rad0,theta0)
        del uinterpfunction, vinterpfunction
        
    ur = uinterp*np.cos(theta0) + vinterp*np.sin(theta0) #radial wind (positive away from center)
    ut = vinterp*np.cos(theta0) - uinterp*np.sin(theta0) #tangential wind (positive counterclockwise)
    del uinterp, vinterp

    return rad0, theta0, ur, ut


####################################
####### ASHRAF'S Functions #########

#################################################################################################

def distance(lat1, lon1, lat2, lon2):
    # Radius of Earth in kilometers
    R = 6371
    g = 9.81
    # Convert degrees to radians using numpy
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    # Haversine formula using numpy functions
    a = np.sin(dLat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Distance in kilometers
    d = R * c
    # Convert to meters
    return d * 1000

#################################################################################################

def calculate_total_flux(delp_values, u_values, v_values, so4_values, 
                         dyright, dyleft, dxhigh, dxlow, g=9.81, dt=1800):
    """
    Optimized version of total flux calculation by parallelizing computations and reducing redundancy.
    """
    flux_east_lev = []
    flux_west_lev = []
    flux_north_lev = []
    flux_south_lev = []
    mass_lev = []

    # Prepare tasks for parallel execution
    tasks = []
    for t in range(int(len(delp_values))):
        # Precompute common terms to avoid redundant calculations
        so4_term = so4_values[t] / g
        
        # Create delayed tasks for each flux calculation
        tasks.append((
            delayed((-1 * delp_values[t] * dyright * u_values[t] * so4_term)
                    .isel(lon=-1).sum(dim=['lat'])),
            delayed((delp_values[t] * dyleft * u_values[t] * so4_term)
                    .isel(lon=0).sum(dim=['lat'])),
            delayed((delp_values[t] * dxhigh * v_values[t] * so4_term)
                    .isel(lat=-1).sum(dim=['lon'])),
            delayed((-1 * delp_values[t] * dxlow * v_values[t] * so4_term)
                    .isel(lat=0).sum(dim=['lon'])),
            delayed((delp_values[t] * dyright * ((dxhigh + dxlow) / 2) * so4_term)
                    .sum(dim=['lon', 'lat']))
        ))

    # Compute the delayed tasks in parallel
    results = dask.compute(*tasks)

    # Separate the results into the appropriate lists
    for res in results:
        flux_east_lev.append(res[0])
        flux_west_lev.append(res[1])
        flux_north_lev.append(res[2])
        flux_south_lev.append(res[3])
        mass_lev.append(res[4])

    # Calculate mass change between time steps
    mass_change_lev = [(mass_lev[t+1] - mass_lev[t]) / dt for t in range(len(mass_lev) - 1)]

    # Ensure that flux lists have the same size as mass_change_lev
    flux_east_lev = flux_east_lev[1:]
    flux_west_lev = flux_west_lev[1:]
    flux_north_lev = flux_north_lev[1:]
    flux_south_lev = flux_south_lev[1:]

    # Calculate the total flux
    total_flux = [m + e + w + n + s for m, e, w, n, s in zip(mass_change_lev, flux_east_lev, flux_west_lev, flux_north_lev, flux_south_lev)]
    
    total_flux_da = xr.DataArray(
        data=total_flux,
        dims=['time', 'lev'],
        coords={
        'time': [total_flux[t].time.values for t in range(len(total_flux))],
        'lev': total_flux[0].lev.values
        },
        name='total_flux_da'
        )


    return total_flux_da

#################################################################################################

def extract_values_with_buffer(dsv, dsu, delp, so4, minplat, minplon, timee, lev, buffer):
    """
    Extracts u, v, delpt, and so4 values for each time step using a buffer region around 
    the minimum pressure latitude and longitude.

    Parameters:
    - dsv: Dataset containing northward wind component ('v')
    - dsu: Dataset containing eastward wind component ('u')
    - delp: DataArray of pressure thickness values
    - so4: DataArray of SO4 concentrations
    - minplat: Array of latitudes with the minimum pressure at each time step
    - minplon: Array of longitudes with the minimum pressure at each time step
    - timee: Array of time indices corresponding to each time step
    - lev: The level(s) to select in the vertical dimension
    - buffer: Degree buffer around the min pressure point (default is 1)

    Returns:
    - u_values: List of selected u values for each time step
    - v_values: List of selected v values for each time step
    - delp_values: List of selected delpt values for each time step
    - so4_values: List of selected so4 values for each time step
    """
    u_values = []
    v_values = []
    delp_values = []
    so4_values = []

    # Loop through each time step
    for t in range(len(minplat)):
        # Define lat/lon selection with buffer space using slice()
        latsel = slice(minplat[t] - buffer, minplat[t] + buffer)
        lonsel = slice(minplon[t] - buffer, minplon[t] + buffer)

        # Select v, u, delpt, and so4 data for the current time step and buffer region
        v = dsv['v'].sel( lev=lev, lat=latsel, lon=lonsel).sel(time=timee[t], method='nearest')
        u = dsu['u'].sel( lev=lev, lat=latsel, lon=lonsel).sel(time=timee[t], method='nearest')
        delpt = delp.sel( lev=lev, lat=latsel, lon=lonsel).sel(time=timee[t], method='nearest')
        so4t = so4.sel( lev=lev, lat=latsel, lon=lonsel).sel(time=timee[t], method='nearest')

        # Append the selected data to lists
        u_values.append(u)
        v_values.append(v)
        delp_values.append(delpt)
        so4_values.append(so4t)

    return u_values, v_values, delp_values, so4_values

#################################################################################################

def compute_side_fluxes(delp_values, u_values, v_values, so4_values, dxlow, dxhigh, dyleft, dyright, g=9.81):
    """
    Computes the fluxes in the east, west, north, and south directions for each time step.

    Parameters:
    - delp_values: List of DataArray objects representing pressure thickness values at each time step
    - u_values: List of DataArray objects representing eastward wind component at each time step
    - v_values: List of DataArray objects representing northward wind component at each time step
    - so4_values: List of DataArray objects representing SO4 concentrations at each time step
    - dxlow: Grid spacing in the x-direction at the southern boundary
    - dxhigh: Grid spacing in the x-direction at the northern boundary
    - dyleft: Grid spacing in the y-direction at the western boundary
    - dyright: Grid spacing in the y-direction at the eastern boundary
    - g: Acceleration due to gravity (default is 9.81 m/s^2)

    Returns:
    - flux_east_lev: List of flux values for the eastern boundary at each time step
    - flux_west_lev: List of flux values for the western boundary at each time step
    - flux_north_lev: List of flux values for the northern boundary at each time step
    - flux_south_lev: List of flux values for the southern boundary at each time step
    """
    flux_east_lev = []
    flux_west_lev = []
    flux_north_lev = []
    flux_south_lev = []

    # Loop through each time step
    for t in range(int(len(delp_values) / 10)):
        # Compute fluxes for each direction
        flux_east = (
            -1 * delp_values[t] * dyright * u_values[t] * so4_values[t] / g
        ).isel(lon=-1).sum(dim=['lat']).compute()

        flux_west = (
            delp_values[t] * dyleft * u_values[t] * so4_values[t] / g
        ).isel(lon=0).sum(dim=['lat']).compute()

        flux_north = (
            -1 * delp_values[t] * dxhigh * v_values[t] * so4_values[t] / g
        ).isel(lat=-1).sum(dim=['lon']).compute()

        flux_south = (
            delp_values[t] * dxlow * v_values[t] * so4_values[t] / g
        ).isel(lat=0).sum(dim=['lon']).compute()

        # Append the computed flux for each time step
        flux_east_lev.append(flux_east)
        flux_west_lev.append(flux_west)
        flux_north_lev.append(flux_north)
        flux_south_lev.append(flux_south)

    return flux_east_lev, flux_west_lev, flux_north_lev, flux_south_lev

#################################################################################################

def compute_mass_and_mass_change(delp_values, so4_values, dxlow, dxhigh, dyright, g=9.81, dt=1800):
    """
    Computes the mass and mass change between consecutive time steps using the provided data arrays.

    Parameters:
    - delp_values: List of DataArray objects representing pressure thickness values at each time step
    - so4_values: List of DataArray objects representing SO4 concentrations at each time step
    - dxlow: Grid spacing in the x-direction at the southern boundary
    - dxhigh: Grid spacing in the x-direction at the northern boundary
    - dyright: Grid spacing in the y-direction at the eastern boundary
    - g: Acceleration due to gravity (default is 9.81 m/s^2)
    - dt: Time step duration in seconds (default is 1800 seconds)

    Returns:
    - mass_lev: List of mass values at each time step
    - mass_change_lev: List of mass change values between consecutive time steps
    """
    mass_lev = []

    # Loop over each time step to compute mass
    for t in range(len(delp_values)):
        mass_levv = (
            delp_values[t] * dyright * ((dxhigh + dxlow) / 2) * so4_values[t] / g
        ).sum(dim=['lon', 'lat']).compute()

        # Append the result for the current time step
        mass_lev.append(mass_levv)

    mass_change_lev = []

    # Loop over each time step to compute mass change
    for t in range(len(mass_lev) - 1):
        mass_change_levv = (mass_lev[t + 1] - mass_lev[t]) / dt
        mass_change_lev.append(mass_change_levv)

    return mass_lev, mass_change_lev

#################################################################################################

####################################
####### Parallel Functions #########

#################################################################################################

def calculate_flux(delp, u, v, so4, dyright, dyleft, dxhigh, dxlow, g_inv):
    flux_east = (-1 * delp * dyright * u * so4 * g_inv).isel(lon=-1).sum(dim=['lat'])
    flux_west = (delp * dyleft * u * so4 * g_inv).isel(lon=0).sum(dim=['lat'])
    flux_north = (-1 * delp * dxhigh * v * so4 * g_inv).isel(lat=-1).sum(dim=['lon'])
    flux_south = (delp * dxlow * v * so4 * g_inv).isel(lat=0).sum(dim=['lon'])
    return flux_east + flux_west + flux_north + flux_south

#################################################################################################

def calculate_mass(delp, so4, dyright, dx_avg, g_inv):
    return (delp * dyright * dx_avg * so4 * g_inv).sum(dim=['lon', 'lat'])

#################################################################################################

def calculate_total_flux_optimized(delp_values, u_values, v_values, so4_values, 
                                   dyright, dyleft, dxhigh, dxlow, g=9.81, dt=1800):
    """
    Corrected and optimized version of the total flux calculation function for list inputs.
    """
    dx_avg = (dxhigh + dxlow) / 2
    g_inv = 1 / g

    total_flux = []
    mass_changes = []
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        futures = []
        for i in range(len(delp_values)-1):
            # Submit tasks for flux and mass calculation for both current and next time step
            futures.append(executor.submit(calculate_flux, delp_values[i], u_values[i], v_values[i], so4_values[i], 
                                           dyright, dyleft, dxhigh, dxlow, g_inv))
            futures.append(executor.submit(calculate_mass, delp_values[i], so4_values[i], dyright, dx_avg, g_inv))
            futures.append(executor.submit(calculate_mass, delp_values[i+1], so4_values[i+1], dyright, dx_avg, g_inv))
        
        # Collect the results in order
        for i in range(len(delp_values)-1):
            flux = futures[3*i].result()  # flux for time step i
            mass_curr = futures[3*i + 1].result()  # mass at time i
            mass_next = futures[3*i + 2].result()  # mass at time i+1
            
            # Calculate mass change and total flux
            mass_change = (mass_next - mass_curr) / dt
            total_flux.append(mass_change + flux)

    # Convert to xarray DataArray
    times = [arr.time.values for arr in delp_values[1:]]
    levels = delp_values[0].lev.values
    total_flux_values = np.array([flux.values for flux in total_flux])

    total_flux_da = xr.DataArray(
        data=total_flux_values,
        dims=['time', 'lev'],
        coords={
            'time': times,
            'lev': levels
        },
        name='total_flux_da'
    )

    return total_flux_da
