#!/usr/bin/env python
# -*- coding: utf-8 -*-
# The original code is found in the following repository. 
# https://github.com/mastnk/imagedegrade
# This code has two modifications compared to the orginal code: useing Numpy RandomState and clipping Gaussian noisy images.

try:
    # python 2
    from StringIO import StringIO as io_memory
except ImportError:
    # python 3
    from io import BytesIO as io_memory

from PIL import Image
import numpy as np

#import imagedegrade.np
from datahandler import imagedegrade_np2

def im2np( im ):
    np = np.asarray( np ).astype( np.float32 )
    np.flags.writeable = True
    return np

def np2im( np ):
    return Image.fromarray( np.uint8(np.clip(0,255)) )

def jpeg( input, jpeg_quality, rs = None, **kwargs ):
    if( not isinstance(input, Image.Image) ):
        msg = 'The input should be Image.Image.'
        raise TypeError( msg )

    buffer = io_memory()
    input.save( buffer, 'JPEG', quality = jpeg_quality, **kwargs )
    buffer.seek(0)
    return Image.open( buffer )

def noise( input, noise_sigma, rs = None ):
    if( not isinstance(input, Image.Image) ):
        msg = 'The input should be Image.Image.'
        raise TypeError( msg )

    array = np.asarray( input ).astype( np.float32 )
    array.flags.writeable = True

    #array = imagedegrade.np.noise( array, noise_sigma, rs )
    array = imagedegrade_np2.noise( array, noise_sigma, rs )

    return Image.fromarray( np.uint8(np.clip(array,0,255)) )

def saltpepper( input, p , rs = None ):
    if( not isinstance(input, Image.Image) ):
        msg = 'The input should be Image.Image.'
        raise TypeError( msg )

    array = np.asarray( input )
    #array = imagedegrade.np.saltpepper( array, p, (0,255), rs )
    array = imagedegrade_np2.saltpepper( array, p, (0,255), rs )
    return Image.fromarray( np.uint8(array) )

def blur( input, blur_sigma, rs = None ):
    if( not isinstance(input, Image.Image) ):
        msg = 'The input should be Image.Image.'
        raise TypeError( msg )

    array = np.asarray( input )
    #array = imagedegrade.np.blur( array, blur_sigma )
    array = imagedegrade_np2.blur( array, blur_sigma )
    return Image.fromarray( np.uint8(array) )