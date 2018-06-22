# -*- coding: utf-8 -*-
from matplotlib.pyplot import *
from collections import namedtuple
import re
import os
import calendar

from ipywidgets import widgets

import pandas as pd
import numpy as np
from scipy import stats

from numpy.random import randint
from IPython.display import HTML

from sklearn.ensemble import ExtraTreesRegressor
from numpy import *

# ========================================================================

# директория с данными
data_dir = './data/'
agg_file_name = '{aggregate_by}agg_{year}-{month:02}.csv'

# ========================================================================

YOUR_API_KEY='AIzaSyANmGkaFLuUPwnD6qwpSZVCSTnL533C1hc'

# ========================================================================

# Количество регрессоров для модели AGGLAR
NHOURS = 6

# ========================================================================

def _g(c,i):
    return c[i] if isinstance(c,(list,tuple)) else c

class LatLong(namedtuple('LatLong', 'lat long')):

    def __add__(self, c):
        return LatLong(*[self[i]+_g(c,i) for i in [0,1]])

    def __sub__(self, c):
        return LatLong(*[self[i]-_g(c,i) for i in [0,1]])
    
    def __mul__(self, c):
        return LatLong(*[1.0*self[i]*_g(c,i) for i in [0,1]])
    
    def __div__(self, c):
        return LatLong(*[1.0*self[i]/_g(c,i) for i in [0,1]])

# ========================================================================

# количество интервалов разбиения по одной оси
N = 50

# координаты региона с которым придется работать
NY = [LatLong(40.49612, -74.25559), LatLong(40.91553, -73.70001)]

# размер одной ячейки
RS = (NY[1]-NY[0])/N

# координаты Empire Street Building
ESB = LatLong(40.748404, -73.985721)

# id зоны (одна из отобранных в предыдущем задании) с которой будем работать в этом задании
# собственно зона с Empire Street Building
RID = 1231 

# ========================================================================
### Регионы (зоны) - нумерация и другие функции

# returns region id by lat lng
def regid_by_ll(ll):
    ll = LatLong(*ll)
    nll = (ll - NY[0])/(NY[1] - NY[0]) * N
    a = lambda x: int(x) if int(x)<N else N-1
    return a(nll.lat) + a(nll.long) * N + 1    

# returns region id by number on lat/lng, where lat/lng is from 1 to N
def regid_by_nll(nll):
    return nll[0] + (nll[1]-1)*N

# returns region rectangle by number on lat/lng, where lat/lng is from 1 to N
def reg_by_nll(nll):
    return [NY[0] + RS * (nll[0]-1, nll[1]-1), NY[0] + RS * (nll[0], nll[1])]

def nll_by_regid(regid):
    return (regid-1)%N + 1, (regid-1)/N + 1


def _sep(aggregate_by): return aggregate_by + '_' if aggregate_by else aggregate_by


# load aggreagated data for year, month and (n-1) months back to past
# if n is 0 load all past data
def load_aggregated(n=-1, year=2016, month=5, aggregate_by=None, verbose = True):
    
    aggregate_by = aggregate_by if aggregate_by is not None else ''
    
    n = n if n>=0 else 1000000 # заведомо большое число файлов
    
    df = None
    for i in range(n):
        afn = agg_file_name.format(year=year, month=month, aggregate_by = _sep(aggregate_by))
        if not os.path.exists(data_dir + afn): break
        if verbose: print 'Loading ', afn, '...', 
        df2 = pd.read_csv(data_dir + afn, index_col=0, parse_dates=True)
        df2.columns = map(int, df2.columns)
        df = df2 if df is None else pd.concat([df2,df])
        if verbose: print 'Done'
        
        month -=1
        if month == 0:
            month = 12
            year -= 1

    return df

# ========================================================================
def basic_template(template, tdict = {}):
    
    def sub(match):
        obj = match.group(1)
        ind = None
        attr = None

        m = re.match('(.*)\[(.*)\]', obj)
        if m:
            obj = m.group(1)
            ind = int(m.group(2))
        else:
            m = re.match('(.*)\.(.*)', obj)
            if m:
                obj = m.group(1)
                attr = m.group(2)

        obj = tdict[obj] if obj in tdict else globals()[obj]
        
        # print "O:", obj, ind, attr

        obj = obj[ind] if ind is not None else getattr(obj, attr) if attr is not None else obj
    
        return str(obj)
    
    return re.sub('{{(.*?)}}', sub, template)


# ========================================================================

def google_map(center = ESB, zoom = 10, size = (400, 400), title = '',
               markers = [], rectangles = [],
               box = NY, box_color = '#FF0000', box_action = '', box_marker = ESB,
               map_action = None,
               verbose = False):
   
    # map_action is called when bounds of map are changed
    # map_action(zoom, center_lat, center_lng)
    
    map_action_html = '''
        map.addListener('bounds_changed', function() {
            c = map.getCenter();
            z = map.getZoom();
            var cmd = '%s(' + z + ',' + c.lat() + ',' + c.lng() + ')';
            IPython.notebook.kernel.execute(cmd);
        });
    
    '''%map_action if map_action else ''
    
    box_template = '''
        var box = new google.maps.Rectangle({
            strokeWeight: 2,
            strokeColor: '{{box_color}}',
            strokeOpacity: 1,
            fillOpacity: 0,
            map: map,
            bounds: {
                south: {{south}},
                west:  {{west}},
                north: {{north}},
                east:  {{east}}
            }
        });
        
        var marker = new google.maps.Marker({
              position: {lat: {{box_marker[0]}}, lng: {{box_marker[1]}} },
              map: map,
        });
        
        box.addListener('dblclick', function(e) {
            marker.setPosition(e.latLng);
            var cmd = '{{box_action}}('+e.latLng.lat()+','+e.latLng.lng()+')';
            IPython.notebook.kernel.execute(cmd);
        });
    '''
    box_html = ''
    if box:
        (south, west), (north, east) = box
        box_html = basic_template(box_template, locals())
    
    marker_template = '''
        new google.maps.Marker({
              position: {lat: {{marker[0]}}, lng: {{marker[1]}} },
              map: map,
        });
    '''    
    markers_html = ''.join([basic_template(marker_template, locals()) for marker in markers])
        
    rect_template = '''
        new google.maps.Rectangle({
            strokeWeight: 0,
            fillColor: '{{color}}',
            fillOpacity: {{opacity}},
            clickable: false,
            map: map,
            bounds: {
                south: {{south}},
                north: {{north}},
                east:  {{east}},
                west:  {{west}}
            }
        });
    '''    
    rect_html = ''.join([
        basic_template(rect_template, locals()) 
        for ((south, west), (north, east)), color, opacity in rectangles
    ])

    
    
    
    template = '''
<!DOCTYPE html>
<html>
  <head>
    <style>
       #map{{map}} {
        height: {{size[1]}}px;
        width: {{size[0]}}px;
       }
    </style>
  </head>
  <body>
    {{title}}
    <div id="map{{map}}"></div>
    <script>
      function initMap() {
        var map = new google.maps.Map(document.getElementById('map{{map}}'), {
          zoom: {{zoom}},
          disableDoubleClickZoom: true, 
          center: {lat: {{center[0]}}, lng: {{center[1]}} }
        });
        
        {{map_action_html}}
        
        {{markers_html}}
        {{rect_html}}
        {{box_html}}
      }
    </script>
    
    
    <script async defer
    src="https://maps.googleapis.com/maps/api/js?key={{YOUR_API_KEY}}&callback=initMap">
    </script>
    
  </body>
</html>
    '''
    if title: title = basic_template('<h4>{{title}}</h4>', locals())
    map = randint(100000)
    
    html = basic_template(template, locals())
    
    if verbose: print html

    return HTML(html)


def gen_arf(ts, nar):
    return pd.DataFrame(
        array([ts.shift(i) for i in range(1,nar+1)]).T, 
        index = ts.index,
        columns = ['AR{:04}'.format(i) for i in range(1,nar+1)]
    )

class LAR:
    def __init__(self, endog, nar = 1):
        self.endog = endog
        self.nar = nar
        self.m = ExtraTreesRegressor(n_estimators=95)

    def fit(self):
        X = gen_arf(self.endog, self.nar)
        self.m.fit(X[self.nar:], self.endog[self.nar:])
        return self
    
    def predict(self, start = 0, end = -1):
        # including end like in SARIMAX!

        n = len(self.endog)
        
        if start < self.nar: start = self.nar
        if end == -1: end = n - 1
        n_after = end - (n - 1) if end > n - 1 else 0
        
        y = self.endog.copy()
        
        dt = y.index[-1] - y.index[-2]
        y = y.append(pd.Series(index=pd.DatetimeIndex(
            start = y.index[-1] + dt, 
            freq = dt, 
            periods = n_after
        )))
        
        y_out = pd.Series(index=y.index)
        
        XE = empty((n + n_after, 0))
            
        start_pred = start if start < n else n
        
        for i in range(start_pred, end+1):
            XA = array([flip( y[i-self.nar : i].values, 0)])
            X = hstack([XA, XE[i:i+1]])
            yy = self.m.predict(X)
            y_out[i] = yy
            if i>=n: y[i] = yy
        
        return y_out[start:end+1]
    
    def update(self, endog):
        '''
        Appends endog and exog without re-fitting model. New data will be used for prognosis.
        Be sure time index is in coherence with existing, no checks inside.
        '''
        
        self.endog = pd.concat([self.endog, endog])
        
    def reset(self, tr):
        '''
        Cut updated endog and exog down to tr length 
        '''
        self.endog = self.endog[:tr]