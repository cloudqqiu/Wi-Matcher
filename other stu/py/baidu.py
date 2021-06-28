# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 21:46:16 2018

@author: py199
"""

import requests
from urllib import request
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


for i in range(0,len(list3)):
    f2=open(list3[i]+".txt","w",encoding="utf-8")
    f=open(list3[i]+".html","wb")
    str1=request.quote(list3[i]) 
    url = "http://www.baidu.com/s?tn=80035161_2_dg&wd="+str1;

    
    req = request.Request(url)
    wbdata = request.urlopen(req).read()
    f.write(wbdata)
    f.close()
    
    print(i)
    soup = BeautifulSoup(wbdata,'lxml')
    print(soup)
    urls = soup.find_all(class_="result c-container")
    print(len(urls))
    for item in urls:
        itemurl=item.find_all(class_="c-abstract")
        if len(itemurl)==0:
            continue
        for i in itemurl:
              f2.write(i.text+"$$$\n")
              print(i.text)
              print("\n\n")
        itemtitle= item.find_all(class_="t")
            
        for j in itemtitle:
            f2.write(j.text+"$$$\n")
            print(j.text)
            print("\n\n")
            soup1 = BeautifulSoup(str(j),'lxml')
            res_tr=r"<a.*?href=\"(.*?)\" target=.*?<\/a>"
            m_tr =  re.findall(res_tr,str(j),re.S|re.M)
            for i in m_tr:
                f2.write(i+"$$$\n\n")
                print(i)
                print("\n\n")
    f2.close()


#特殊数据单独爬取
'''
list5=["😳😳柑鹿-Manna","😳😳柑鹿MANNA💤"]
for i in range(0,len(list5)):
    f1=open("柑鹿MANNA("+str(i+1)+").txt","w",encoding="utf-8")
    f=open("柑鹿MANNA("+str(i+1)+").html","wb")
    str1=request.quote(list5[i]) 
    url = "http://www.baidu.com/s?tn=80035161_2_dg&wd="+str1;

    
    req = request.Request(url)
    wbdata = request.urlopen(req).read()
    f.write(wbdata)
    f.close()
    
    print(i)
    soup = BeautifulSoup(wbdata,'lxml')

    urls = soup.find_all(class_="result c-container")
    print(len(urls))
    for item in urls:
        itemurl=item.find_all(class_="c-abstract")
        if len(itemurl)==0:
            continue
        for i in itemurl:
              f1.write(i.text+"$$$\n")
              print(i.text)
              print("\n\n")
        itemtitle= item.find_all(class_="t")
            
        for j in itemtitle:
            f1.write(j.text+"$$$\n")
            print(j.text)
            print("\n\n")
            soup1 = BeautifulSoup(str(j),'lxml')
            res_tr=r"<a.*?href=\"(.*?)\" target=.*?<\/a>"
            m_tr =  re.findall(res_tr,str(j),re.S|re.M)
            for i in m_tr:
                f1.write(i+"$$$\n\n")
                print(i)
                print("\n\n")
    f1.close()
'''


