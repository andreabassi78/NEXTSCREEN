# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 15:35:26 2022

@author: Andrea Bassi
"""

from magicgui import magicgui
import datetime
import pathlib
import os
    
@magicgui(
    call_button="Print filename",
    slider_float={"widget_type": "FloatSlider", 'max': 10},
    dropdown={"choices": ['first', 'second', 'third']},
)
def my_ui(
    maybe: bool,
    some_int: int,
    spin_float: float=3.14159,
    slider_float=4.5,
    string="Text goes here",
    dropdown='first',
    date=datetime.datetime.now(),
    filename=pathlib.Path(os.getcwd())
            ):
    
    print('The name of the selected file is:',filename)
    my_ui.string.value = 'here we are'
    return(some_int*3)
    
my_ui.show(run=True)

    