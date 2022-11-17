import glob
import os
from re import I
import sys

from joblib import Parallel, delayed
import tifffile
import numpy as np
import pandas as pd

vrange = np.arange(256, dtype=np.uint8)
lut_colors = {
    'magenta' : np.stack([vrange, 0*vrange, vrange]),
    'green' : np.stack([0*vrange, vrange, 0*vrange]),
    'yellow' : np.stack([1*vrange, 1*vrange, 0*vrange]),
    'cyan' : np.stack([0*vrange, 1*vrange, 1*vrange]),
    'red' : np.stack([1*vrange, 0*vrange, 0*vrange]),
    'blue' : np.stack([0*vrange, 0*vrange, vrange]),
    'gray' : np.stack([vrange, vrange, vrange]),
}

if sys.platform == 'win32':
    fiji = "U:/Fiji/Fiji.app/ImageJ-win64.exe"
elif sys.platform == 'darwin':
    fiji = "/Volumes/projects/Fiji/Fiji.app/Contents/MacOS/ImageJ-macosx"
else:
    fiji = "/n/projects/Fiji/Fiji.app/ImageJ-linux64"

    
if sys.platform == 'win32':
    correction_image_file = 's:/micro/cry/ww2630/cor_ref.tif'
elif sys.platform == 'linux':
    correction_image_file = '/n/core/micro/cry/ww2630/cor_ref.tif'
elif sys.platform == 'darwin':
    correction_image_file = '/Volume/core/micro/cry/ww2630/cor_ref.tif'
else:
    correction_image_file = None
    
if correction_image_file is not None:
    correction_image = tifffile.imread(correction_image_file)
else:
    correction_image = [None]

'''
r04c01f04p01-ch1sk1fk1fl1.tiff'
0123456789012345
          111111
          
r - row
c - column
f - field
p - z slice
'''

def get_image_files(screen_dir, image_glob='.tiff'):
    image_files = sorted(os.listdir(screen_dir))
    image_files = [x for x in image_files if x.endswith(image_glob)]
    prefixes = set([x.split('-')[0][:-3] for x in image_files])
    fdict = {k:[] for k in prefixes} 

    pzmax = 0
    chmax = 0
    for f in image_files:
        k = f.split('-')[0][:-3]
        dpos = f.index('-')
        rowstart = f.index('r') + 1
        colstart = f.index('c') + 1
        fieldstart = f.index('f') + 1
        pstart = f.index('p')  + 1
        row = int(f[rowstart:colstart - 1])
        col = int(f[colstart:fieldstart - 1])
        field = int(f[fieldstart:pstart - 1])
        pz = int(f[pstart:dpos]) - 1

        channelindex = f.index('ch') + 2
        ch = int(f[channelindex]) - 1
        fdict[k].append(f)
        if ch > chmax:
            chmax = ch

        if pz > pzmax:
            pzmax = pz

    return fdict, chmax + 1, pzmax + 1

def get_image(fdir, imagefiles, nchannels, nz, bin=1, project='MAX'):

    stack = None
    for f in imagefiles:
        k = f.split('-')[0][:-3]
        dpos = f.index('-')
        rowstart = f.index('r') + 1
        colstart = f.index('c') + 1
        fieldstart = f.index('f') + 1
        pstart = f.index('p')  + 1
        row = int(f[rowstart:colstart - 1])
        col = int(f[colstart:fieldstart - 1])
        field = int(f[fieldstart:pstart - 1])
        pz = int(f[pstart:dpos])

        channelindex = f.index('ch') + 2
        ch = int(f[channelindex]) - 1

        data = tifffile.imread(f"{fdir}/{f}")
        if bin == 2:
            _dr = data.reshape(
                (data.shape[0]//2, 2, data.shape[1]//2, 2))
            data = _dr.sum(axis=-1).sum(axis=1)
            
        sy, sx = data.shape
        if stack is None:
            stack = np.zeros((nz, nchannels, sy, sx), dtype=data.dtype)
        stack[pz - 1, ch, :, : ] = data

    if project=='MAX':
        stack = np.squeeze(stack.max(axis=0, keepdims=True))
    else:
        stack = np.squeeze(stack)
    #stack = np.expand_dims(stack, 0)
    resdict = {'files':imagefiles,
               'row':row,
               'col':col,
               'field':field,
               'data':stack}

    return resdict 

def image_generator(fdir, image_glob='.tiff', bin=1):
    fdict, nchannels, nz = get_image_files(fdir)
    for k, v in fdict.items():
        yield get_image(fdir, k, v, nchannels, nz, bin=bin)
        

def correct_flatness(img, cor_img):
    '''
    Using Sean McKinney's Correct_Flatness
    https://github.com/jouyun/smc-plugins/blob/master/src/main/java/splugins/Correct_Flatness.java
     - DoCorrectUsingProvided
     
     - find the max of the corrected image
     - normalize the correction image by the max
     
     - (original_image - 90)/normalized_correction_image
     
     - using 90 as the background
     
    '''
    
    cmax = cor_img.max()
    n_cor_img = cor_img/cmax
    corrected = (img - 90)/n_cor_img
    
    return corrected
    
def process(s, saveto, data, nchannels, cor_img):
    filename = f"{saveto}/{s}.tif" 
    colors = list(lut_colors.keys())[:nchannels]
    luts = [lut_colors[c] for c in colors]
    
    if cor_img is not None:
        data = correct_flatness(data, cor_img)
    tifffile.imwrite(filename, data,
            imagej=True,
            photometric='minisblack',
            metadata={'axes': 'CYX',
                      'Composite mode':'composite',
                      'LUTs':luts})
    #tifffile.imwrite(filename, data)

def imwrite(filename, data, axes="CYX", resolution=None, spacing=1,
            colors=None):

    nchannels = len(data)
    if colors is None:
        colors = list(lut_colors.keys())[:nchannels]

    luts = [lut_colors[c] for c in colors]
    
    if resolution is None:
        resolution = np.ones((len(data.shape),))
    
    tifffile.imwrite(filename, np.squeeze(data),
            imagej=True,
            photometric='minisblack',
            resolution=resolution,
            metadata={'axes': axes,
                      'spacing': spacing,
                      'Composite mode':'composite',
                      'LUTs':luts})
    
def pfunc(d, filenames, **kwargs):
    fdir = kwargs['folder']
    nz = kwargs['nz']
    nchannels = kwargs['nchannels']
    saveto = kwargs['saveto']
    cor_img = kwargs['correction_image']
    
    image_dict = get_image(fdir, filenames, nchannels, nz, bin=1)
    _ = process(d, saveto, image_dict['data'], nchannels, cor_img)

    return

def run(folders, saveto, globpattern='*.tiff', njobs=2,
        colors=None, correction_image=None):
    
    if isinstance(folders, str):
        folders = [folders]

    if not os.path.exists(saveto):
        try:
            os.makedirs(saveto)
        except:
            print(f"Can't create: {saveto}\n. Exiting")

    print(folders)
    kwargs = {'globpattern':globpattern}
    for i, folder in enumerate(folders):
        print(folder)
        imagefiles, nchannels, nz = get_image_files(folder)

        kwargs['folder'] = folder
        kwargs['nz'] = nz
        kwargs['nchannels'] = nchannels
        kwargs['saveto'] = saveto
        kwargs['correction_image'] = correction_image
       
        if njobs < 0: 
            i = 1 
            n = -njobs 
            for k, v in imagefiles.items():
                _ = pfunc(k, v, **kwargs)
                if i >= n:
                    break
                i += 1
        else:
            _ = Parallel(n_jobs=njobs)\
                  (delayed(pfunc)(k, v,  **kwargs) for k, v in imagefiles.items())
            
            
def dataframe_from_xml(xmldata):
    #coords_list= list()

    images = xmldata.getElementsByTagName("Images")[0]

    image_dict = {
            "filename":[],
            "row":[],
            "col":[],
            "well":[],
            "field":[],
            "channel":[],
            "plane":[],
            "posx":[],
            "posy":[],
            "posz":[],
            "width":[],
            "height":[],
            "pixel_size":[],
    }
    rowmap = '_abcdefghijklmnop'
    for i, x in enumerate(images.childNodes):

        if x.nodeType == 1:
            node_posx = x.getElementsByTagName("PositionX")[0]
            node_posy = x.getElementsByTagName("PositionY")[0]
            node_posz = x.getElementsByTagName("PositionZ")[0]
            posx = 1e3*float(node_posx.firstChild.nodeValue) # + .1*np.random.rand()
            posy = 1e3*float(node_posy.firstChild.nodeValue) # + .1*np.random.rand()
            posz = 1e3*float(node_posz.firstChild.nodeValue) # + .1*np.random.rand()

            node_ch = x.getElementsByTagName("ChannelID")[0]
            ch = node_ch.firstChild.nodeValue

            node_plane = x.getElementsByTagName("PlaneID")[0]
            plane = int(node_plane.firstChild.nodeValue)
            field = int(x.getElementsByTagName("FieldID")[0].firstChild.nodeValue)
            row = int(x.getElementsByTagName("Row")[0].firstChild.nodeValue)
            col = int(x.getElementsByTagName("Col")[0].firstChild.nodeValue)
            filename = x.getElementsByTagName("URL")[0].firstChild.nodeValue

            sizeX = int(x.getElementsByTagName("ImageSizeX")[0].firstChild.nodeValue)
            sizeY = int(x.getElementsByTagName("ImageSizeY")[0].firstChild.nodeValue)

            pixelsize = 1e3*float(x.getElementsByTagName("ImageResolutionY")[0].firstChild.nodeValue)

            well = f"{rowmap[row]}{col:02d}"
            image_dict["filename"].append(filename)
            image_dict["field"].append(field)
            image_dict["row"].append(row)
            image_dict["col"].append(col)
            image_dict["well"].append(well)
            image_dict["channel"].append(ch)
            image_dict["plane"].append(plane)
            image_dict["posx"].append(posx)
            image_dict["posy"].append(posy)
            image_dict["posz"].append(posz)
            image_dict["width"].append(sizeX)
            image_dict["height"].append(sizeY)
            image_dict['pixel_size'].append(pixelsize)


            #coords_list.append([posz, posy, posx])


    #coords = np.array(coords_list)
    df = pd.DataFrame(image_dict)
    return df

def field_stack(_df, image_path, cor_image, savedir, project, well, field, colors):
    
        if project not in ["MAX", "SUM", "MEAN"]:
            project = None
        h, w = int(_df.iloc[0].height), int(_df.iloc[0].width)
        nchannels = len(_df.channel.unique())
        nz = len(_df.plane.unique())
        stack = np.zeros((nz, nchannels, h, w))
       
        for row in _df.itertuples():
            fname = f"{image_path}/{row.filename}"
            plane = int(row.plane) - 1
            ch = int(row.channel) - 1
            x = tifffile.imread(fname)
            stack[plane, ch] = x
        
        sname = f"{well}_{field:02d}.tif"
        
        zp = 1000*_df.groupby("plane").posz.agg("mean").diff().mean()
        xp = 1000*_df.pixel_size.mean()
        resolution = (1/xp, 1/xp)
        if nchannels > 1:
            cstr = "C"
        else:
            cstr = ""
            
        if nz > 1 and project is None:
            zstr = "Z"
        else:
            zstr = ""
        
        axes = zstr + cstr + "YX"
        
        if project=='MAX':
            pstack = stack.max(axis=0)
        elif project=='SUM':
            pstack = stack.sum(axis=0)
        else:
            pstack = stack
        
        if len(cor_image) > 0:
            cstack = correct_flatness(pstack, cor_image)
        else:
            cstack = pstack

        cstack = cstack.astype(np.uint16)
        imwrite(f"{savedir}/{sname}", cstack, axes,
                resolution=resolution, spacing=zp, colors=colors)
        return sname
    
class Coords:
    
    def __init__(self, rect):
        _x1 = rect[0]
        _x2 = rect[2]
        
        _y1 = rect[1]
        _y2 = rect[3]
        
        self.xmin = min(_x1, _x2)
        self.xmax = max(_x1, _x2)
        
        self.ymin = min(_y1, _y2)
        self.ymax = max(_y1, _y2)

def norm_pos(_x, ngrid, reverse):
    if reverse:
        x = -_x
    else:
        x = _x
    x = x - x.min()
    nx = (x - x.min())/(x.max() - x.min())
    nx = (ngrid - 1)*nx
    nx = nx.round().astype(np.int32)
    return nx


def crop_to_rect(selection, df):
    rc = Coords(selection[-1])
    topdf = df.loc[(df.posx > rc.xmin) & (df.posx < rc.xmax) & (df.posy > rc.ymin) & (df.posy < rc.ymax)].copy()
    fielddf = topdf.groupby("field").agg("mean").reset_index()
    
    bx = len(fielddf.posx.unique())
    by = len(fielddf.posy.unique())
    
    topdf['normx'] = topdf.posx.transform(norm_pos, 0, bx, False)
    topdf['normy'] = topdf.posy.transform(norm_pos, 0, by, True)
    topdf['fnum'] = bx*topdf['normy'] + topdf['normx']
    xoffset = np.mean(np.diff(np.sort(topdf.posx.unique()))/topdf.pixel_size.mean())
    yoffset = np.mean(np.diff(np.sort(topdf.posy.unique()))/topdf.pixel_size.mean())
    topdf['xoffset'] = xoffset*topdf.normx
    topdf['yoffset'] = yoffset*topdf.normy
    
    return topdf, fielddf

def crop_to_rect_bokeh(source, bdf, df):
    '''
    selection will be the indices from the dataframe, bdf
    can get the fields, then get the fields from the main df
    '''
    indices = source.selected.indices
 
    fields = list(bdf.iloc[indices].field.unique())
    topdf = df.loc[df.field.isin(fields)].copy()

    #topdf = df.loc[(df.posx > rc.xmin) & (df.posx < rc.xmax) & (df.posy > rc.ymin) & (df.posy < rc.ymax)].copy()
    fielddf = topdf.groupby("field").mean(numeric_only=True).reset_index()
    
    bx = len(fielddf.posx.unique())
    by = len(fielddf.posy.unique())
    
    print(bx, by)
    topdf['normx'] = topdf.posx.transform(norm_pos, 0, bx, False)
    topdf['normy'] = topdf.posy.transform(norm_pos, 0, by, True)
    topdf['fnum'] = bx*topdf['normy'] + topdf['normx']
    xoffset = np.mean(np.diff(np.sort(topdf.posx.unique()))/topdf.pixel_size.mean())
    yoffset = np.mean(np.diff(np.sort(topdf.posy.unique()))/topdf.pixel_size.mean())
    topdf['xoffset'] = xoffset*topdf.normx
    topdf['yoffset'] = yoffset*topdf.normy
    
    return topdf, fielddf


def getrect(_bdf, source):
    x1 = _bdf.iloc[source.selected.indices].posx.min()
    x2 = _bdf.iloc[source.selected.indices].posx.max()

    y1 = _bdf.iloc[source.selected.indices].posy.min()
    y2 = _bdf.iloc[source.selected.indices].posy.max()

    rect = [x1, y1, x2, y2]
    return rect
    