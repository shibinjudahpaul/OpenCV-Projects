# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 19:20:56 2021

@author: Shibin Judah Paul
"""

import os
import shutil

src = r"C:\\Users\\Shibin Paul\\Documents\\AI-ML projects\\Covid 19 detection\\Unhealthy_dataset"
dstn = r"C:\\Users\\Shibin Paul\\Documents\\AI-ML projects\\Covid 19 detection\\Covid_dataset"

with open('img_list.txt','r') as imagelist:
    readr = imagelist.readlines()
    for names in readr:
        #print(os.path.abspath(os.path.join(src, names)), sep='\n')
        shutil.copy(os.path.abspath(os.path.join(src, names)).replace('\n', ''), os.path.abspath(dstn))
        