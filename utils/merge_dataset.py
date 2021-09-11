# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 09:02:40 2021

@author: jorge
"""


import glob
import shutil
import os

src_dir = "pk_database/pk_database/test/*/"
dst_dir = "pk_database/pk_database/merge_test/"

# for jpgfile in glob.iglob(os.path.join(src_dir, "*.txt")):
#     shutil.copy(jpgfile, dst_dir)

    

with open(dst_dir+'groundtruth.txt', 'w') as outfile:
    for fname in glob.iglob(os.path.join(src_dir, "*.txt")):
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
                
                
    
# src_dir = "pk_database/pk_database/training/*"
# dst_dir = "pk_database/pk_database/merge_train/"
# for jpgfile in glob.iglob(os.path.join(src_dir, "*.txt")):
#     shutil.copy(jpgfile, dst_dir)
    
    
# src_dir = "pk_database/pk_database/test/*"
# dst_dir = "pk_database/pk_database/merge_test/"
# for jpgfile in glob.iglob(os.path.join(src_dir, "*.txt")):
#     shutil.copy(jpgfile, dst_dir)