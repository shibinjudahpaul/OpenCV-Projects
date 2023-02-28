# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 20:15:56 2021

@author: Shibin Judah Paul
"""

import csv


with open('metadata.csv', 'r', encoding='utf-8')as fin, open ('outfile.csv','w', encoding='utf-8') as fout:
    writr = csv.writer(fout)
    readr = csv.reader(fin)
    fields = next(readr)
    #print(fields[23]) 
    #print(fields[18])
    writr.writerow(fields)
    for row in csv.reader(fin):
        if row[4] == 'Pneumonia/Viral/COVID-19' and row[18] == 'PA':
            writr.writerow(row)
            

        
            