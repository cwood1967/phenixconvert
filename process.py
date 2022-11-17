import os

import numpy as np
from skimage.transform import resize

from xml.dom.minidom import parse, parseString
from plotly import express as px

from tqdm import tqdm

import phenix
from phenix import fiji
from phenix import correction_image  


def xml_to_grid(image_path, saveto):
    
    pxml_file = f"{image_path}/Index.idx.xml"
    print(pxml_file)
    xmldata = parse(pxml_file)
    
    df = phenix.dataframe_from_xml(xmldata)
    bdf = df.groupby(["well", "field"]).mean(numeric_only=True).reset_index()
    print(bdf) 
    fig = px.scatter(bdf, x="posx", y="posy", text='field', height=600)
    fig.update_traces(textfont={'size':10})
    fig.update_xaxes(
        scaleanchor='y',
        scaleratio=1
    )
    
    return fig


def convert(image_path, savedir, projection):

    pxml_file = f"{image_path}/Index.idx.xml"
    xmldata = parse(pxml_file)
    
    df = phenix.dataframe_from_xml(xmldata)
    bdf = df.groupby(["well", "field"]).mean(numeric_only=True).reset_index()

    nchannels = int(df.channel.max())
    if nchannels < len(correction_image):
        xcor = correction_image[:nchannels]
    else:
        xcor = correction_image
        
    sizex = int(bdf.width.mean())
    sizey = int(bdf.height.mean())
    
    xcor = resize(xcor, (nchannels, sizex, sizey))
    
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    fg = df.groupby(["field", 'well'])

    colors = ["green", "magenta", "yellow", "blue"]
    for f, gdf in tqdm(fg):
        _ = phenix.field_stack(gdf, image_path, xcor, savedir,
                                      projection, f[1], f[0], colors=colors) 
    


    
    

