# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 19:21:35 2021

@author: Shibin Judah Paul
"""
import csv
import sys



with open('outfile.csv', 'r', encoding='utf-8') as covid_list, open('img_liiist.txt','w',encoding='utf-8') as covid_img_list:
    readr = csv.reader(covid_list)
    writr = csv.writer(covid_img_list)
    fields = next(readr)
    writr.writerow(fields[23])
    for i, row in enumerate(readr):
        try:
              print(row[23])
            # writr.writerow(str(row[23]))
        except IndexError:
            pass
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # # writr.writerow(row[23])
        # # if row[23] == "filename":
        # #     print(row23)
        # size = sys.getsizeof(row)
        # if size > 50:
        #     print(row)
        
    
          