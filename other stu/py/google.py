# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 21:46:16 2018

@author: py199
"""

import requests
from  urllib import request
import re
import time
from bs4 import BeautifulSoup
#from requests_html import HTMLSession



list1=[]
list2=[];
with open('shuju1.txt', 'r',encoding='utf-8-sig') as f1:
    list1 = f1.readlines()
with open('shuju2.txt', 'r',encoding='utf-8-sig') as f2:
    list2 = f2.readlines()
list3=[];
list4=[];
for i in range(0,len(list1)):
    list3.append(list1[i].rstrip('\n'))
print(list3)

for i in range(0,len(list2)):
    list4.append(list2[i].rstrip('\n'))
print(list4)
print(len(list3))



for k in range(0,len(list3)):
    a=list3[k]
    url = "https://www.google.com.hk/search?newwindow=1&safe=strict&ei=TuSaXMioNPCUr7wPqMGkoAE&q="+a+"&oq="+a+"&gs_l=psy-ab.3..0l10.1367202.1370700..1370895...0.0..0.102.555.4j2......0....1..gws-wiz.....0..0i67j0i12j0i7i30.AIsOX1yCwFo"
    f=open(a+".html","w",encoding='utf-8')
    f1=open(a+".txt","w",encoding='utf-8')
    wbdata = requests.get(url).text
    time.sleep(1)
    f.write(wbdata)
    f.close()
    soup = BeautifulSoup(wbdata,'lxml')

    urls = soup.find_all(class_="g")
    print(len(urls))
    for item in urls:
            if(str(item).find("Video for")>=0):
                continue
            
            itemurl=item.find_all(class_="hJND5c")
            if len(itemurl)==0:
                continue
            
            #for j in itemurl:
            soup1 = BeautifulSoup(str(item),'lxml')
                #res_tr = r'<cite>(.*?)</cite>'
                #m_tr =  re.findall(res_tr,str(j),re.S|re.M)
            m_tr = soup1.find_all('cite')
            for i in m_tr:
                f1.write(i.text+"$$$\n")
                print(i.text)
                print("\n\n")
            itemtitle= item.find_all(class_="r")
            
            for i in itemtitle:
                f1.write(i.text+"$$$\n")
                print(i.text)
                print("\n\n")
            
            itemtext=item.find_all(class_="s")
            for i in itemtext:
                print(i.text)
                f1.write(i.text+"$$$\n\n")
                print("\n\n")
    f1.close()


#特殊值爬取
'''
#a="😳😳柑鹿MANNA💤"

a="bbtreewifi"
url = "https://www.google.com.hk/search?newwindow=1&safe=strict&ei=TuSaXMioNPCUr7wPqMGkoAE&q="+a+"&oq="+a+"&gs_l=psy-ab.3..0l10.1367202.1370700..1370895...0.0..0.102.555.4j2......0....1..gws-wiz.....0..0i67j0i12j0i7i30.AIsOX1yCwFo"


wbdata = requests.get(url).text

f=open(a+".html","w",encoding='utf-8')
f1=open(a+".txt","w",encoding='utf-8')
f.write(wbdata)
print(wbdata)
soup = BeautifulSoup(wbdata,'lxml')

urls = soup.find_all(class_="g")


for item in urls:
    if(str(item).find("Video for")>=0):
        continue
            
    itemurl=item.find_all(class_="hJND5c")
    if len(itemurl)==0:
        continue        
            
            #for j in itemurl:
    soup1 = BeautifulSoup(str(item),'lxml')
                #res_tr = r'<cite>(.*?)</cite>'
                #m_tr =  re.findall(res_tr,str(j),re.S|re.M)
    m_tr = soup1.find_all('cite')
    for i in m_tr:
        f1.write(i.text+"$$$\n")
        print(i.text)
        print("\n\n")
    itemtitle= item.find_all(class_="r")
            
    for i in itemtitle:
        f1.write(i.text+"$$$\n")
        print(i.text)
        print("\n\n")
            
    itemtext=item.find_all(class_="s")
    for i in itemtext:
        print(i.text)
        f1.write(i.text+"$$$\n\n")
        print("\n\n")
f1.close()
'''


