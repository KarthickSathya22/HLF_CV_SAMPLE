# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:52:46 2020

@author: ESFITA-USER
"""

import requests

url = 'http://127.0.0.1:5000/file-upload'


file_name = '115503069_1583398644610.pdf'

files = {'file':open(file_name,'rb')}

r = requests.post(url,files=files)

print(r.status_code)

target = file_name.split('.')[0]+'.csv' 

with open(target, 'w') as f:
    f.write(r.text)
