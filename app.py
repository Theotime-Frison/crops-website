import streamlit as st
import numpy as np
import rasterio
from rasterio.plot import show
from tempfile import NamedTemporaryFile
import os
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.express as px
import folium
from streamlit_folium import st_folium, folium_static
import streamlit.components.v1 as components
from shapely.geometry import Point, Polygon
import requests
from PIL import Image
import io
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib as mpl
import gzip
import altair as alt




st.set_page_config(layout = 'wide')

'''
# Satellite Crops

# 🌱 What's growing in my field?
'''

mapping = {0: 'Background',
1: 'Wheat',
 2: 'Hemp',
 3: 'Rapeseed',
 4: 'Cereals (other)',
 5: 'Vegetables',
 6: 'Maize',
 7: 'Oilseeds',
 8: 'Barley',
 9: 'Potatoes',
 10: 'Meadow',
 11: 'Protein crops',
 12: 'Sunflower',
 13: 'Orchards',
 14: 'Beetroot'}

yields = {0: 0.1,
1: 6.325,
 2: 0.1,
 3: 0.1,
 4: 0.1,
 5: 0.1,
 6: 8.7,
 7: 0.1,
 8: 0.1,
 9: 0.1,
 10: 6.95,
 11: 0.1,
 12: 0.1,
 13: 0.1,
 14: 0.1}

def generate_colors_from_cmap(cmap_name, num_colors):
    return [plt.get_cmap(cmap_name)(i) for i in np.linspace(0, 1, num_colors)]

st.set_option('deprecation.showfileUploaderEncoding', False)


# Define columns
c1, c2 = st.columns((1, 1))


# Load images and geoDF


X1 = np.load(gzip.GzipFile('X1.npy.gz'))
X2 = np.load(gzip.GzipFile('X2.npy.gz'))
X = np.concatenate([X1,X2], axis=0)

gdf = gpd.read_file('dataframe.gpkg')
y = np.load(gzip.GzipFile('my_labels.npy.gz'))

X = np.moveaxis(X, 1, 3)

# Define map
m = folium.Map(location=[44, -0.6], zoom_start=9)
m.add_child(folium.LatLngPopup())

folium.GeoJson(
    data=gdf["geometry"],
    highlight_function= lambda feat: {'fillColor': 'red'}
).add_to(m)

def get_pos(lat,lng):
    return lat,lng

with c1 :
    map = st_folium(m, height=600, width=700)

# Define image
with c2 :
    try :
        lat, lng = get_pos(map['last_clicked']['lat'],map['last_clicked']['lng'])
        pnt = Point(lng,lat)

        gdf = gdf.to_crs(4326)

        for i in range(len(gdf)):
            if pnt.within(gdf.geometry[i]):
                my_id = i

        # Predictions
        url = 'https://sat-api-01-msjthztcna-od.a.run.app/predict'
        params = {'eopatch_id' : my_id}

        response = np.array(requests.get(url, params=params).json()['img']).argmax(axis=2)

        # Plot predictions and image
        '''## Prediction'''
        fig, ax = plt.subplots(figsize=(15,15))

        rgb_color = (15/255, 17/255, 22/255)

        fig.patch.set_facecolor(rgb_color)

        categories = [mapping[i] for i in np.unique(response)]

        colors = generate_colors_from_cmap('viridis', len(categories))

        # Set first color totally transparent
        l = [color for color in colors[0]]
        l[3] = 0
        colors[0] = tuple(l)

        cmap= ListedColormap(colors)

        cax = ax.imshow(X[my_id])
        cax = ax.imshow(response, cmap=cmap, alpha=0.6)

        # Legend
        # Create a legend for the categorical data
        patches = [Patch(color=colors[i], label=categories[i]) for i in range(len(categories))]
        legend = ax.legend(handles=patches, loc='upper right',bbox_to_anchor=(1.6, 1), title='Categories', facecolor=rgb_color, edgecolor='none', title_fontsize='20', fontsize='18')
        plt.setp(legend.get_texts(), color='white')
        plt.setp(legend.get_title(), color='white')

        fig.subplots_adjust(right=0.6)

        ax.set_xticks([])
        ax.set_yticks([])

        buf = io.BytesIO()
        plt.savefig(buf, bbox_inches='tight', format='png')
        buf.seek(0)

        st.image(buf, use_column_width=True)



    except:
        my_id = np.random.randint(0, X.shape[0])
        st.image(X[my_id], width=600)


c1, c2, c3 = st.columns((1, 1, 1))



# COMPARISONS ---------------------------------

with c1 :
    try :
        fig, ax = plt.subplots(figsize=(15,15))

        rgb_color = (15/255, 17/255, 22/255)

        fig.patch.set_facecolor(rgb_color)

        cax = ax.imshow(X[my_id])

        ax.set_xticks([])
        ax.set_yticks([])

        buf = io.BytesIO()
        plt.savefig(buf, bbox_inches='tight', format='png')
        buf.seek(0)

        st.image(buf, caption = 'Original image',use_column_width=True)
    except:
        pass

with c2 :
    try :
        fig, ax = plt.subplots(figsize=(15,15))

        rgb_color = (15/255, 17/255, 22/255)

        fig.patch.set_facecolor(rgb_color)

        categories = [mapping[i] for i in np.unique(response)]

        colors = generate_colors_from_cmap('viridis', len(categories))

        # Set first color totally transparent
        l = [color for color in colors[0]]
        l[3] = 0
        colors[0] = tuple(l)

        cmap= ListedColormap(colors)
        cax = ax.imshow(response, cmap=cmap)

        ax.set_xticks([])
        ax.set_yticks([])

        buf = io.BytesIO()
        plt.savefig(buf, bbox_inches='tight', format='png')
        buf.seek(0)

        st.image(buf, caption = 'Prediction',use_column_width=True)
    except:
        pass

with c3 :
    try :
        fig, ax = plt.subplots(figsize=(15,15))

        rgb_color = (15/255, 17/255, 22/255)

        fig.patch.set_facecolor(rgb_color)

        categories = [mapping[i] for i in np.unique(response)]

        colors = generate_colors_from_cmap('viridis', len(categories))

        # Set first color totally transparent
        l = [color for color in colors[0]]
        l[3] = 0
        colors[0] = tuple(l)

        cmap= ListedColormap(colors)
        cax = ax.imshow(y[my_id], cmap=cmap)

        ax.set_xticks([])
        ax.set_yticks([])

        buf = io.BytesIO()
        plt.savefig(buf, bbox_inches='tight', format='png')
        buf.seek(0)

        st.image(buf, caption = 'Ground truth',use_column_width=True)
    except:
        pass



c1, c2 = st.columns((1, 1))
c1.title('🌾 Surface prediction (ha)')
c2.title('🧑‍🌾 Production (t/year)')

with c1 :
    try:
        unique, count = np.unique(response, return_counts=True)
        df = pd.DataFrame({'id': unique, 'count': count})
        df['Crop'] = df.id.map(mapping)
        df['(ha)'] = round(df['count'] * 10 * 10 / 10000, 1)
        df.drop(columns=['count', 'id'],inplace=True)
        df.set_index('Crop', inplace=True)

        chart = alt.Chart(df[df.index != 'Background'].reset_index()).mark_bar().encode(
        x=alt.X('Crop:N', axis=alt.Axis(labelAngle=0)),
        y="(ha):Q"
        ).configure_axis(labelFontSize=15)

        st.altair_chart(chart, theme="streamlit", use_container_width=True)

        #st.bar_chart(df[df.index != 'Background'])
    except:
        st.write('Please select a tile on the map to get prediction.')


with c2 :
    try:
        yields = pd.DataFrame({'id': yields.keys(), 'yields':yields.values()})
        df = pd.DataFrame({'id': unique, 'count': count})

        df['Crop'] = df.id.map(mapping)
        merged_df = pd.merge(df[df.index != 'Background'],yields,on='id', how='inner')

        merged_df['Surface (ha)'] = round(merged_df['count'] * 10 * 10 / 10000, 1)
        merged_df['(t/year)'] = merged_df['Surface (ha)'] * merged_df['yields']

        res_df = merged_df.drop(columns=['count', 'id', 'yields', 'Surface (ha)']).set_index('Crop')

        chart = alt.Chart(res_df[res_df.index != 'Background'].reset_index()).mark_bar().encode(
        x=alt.X('Crop:N', axis=alt.Axis(labelAngle=0)),
        y="(t/year):Q"
        ).configure_axis(labelFontSize=15)

        st.altair_chart(chart, theme="streamlit", use_container_width=True)
    except:
        st.write('Please select a tile on the map to get prediction.')
