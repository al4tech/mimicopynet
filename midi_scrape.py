#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 02:37:33 2017

@author: marshi
"""

import urllib
import urllib.request # 追加
from bs4 import BeautifulSoup
import os
import re

def alpha_pagenum(alpha):
    url = 'http://www.midiworld.com/files/%s/all/'%alpha
    response = urllib.request.urlopen(url)
    soup = BeautifulSoup(response, "lxml")
    listing = soup.find("ul",class_="listing")
    l = len(listing.contents)
    if l <= 4:
        return 1
    else:
        return l-4

def alpha_artists(alpha):
    n = alpha_pagenum(alpha)
    artists = []
    for i in range(n):
        url = 'http://www.midiworld.com/files/%s/all/%d'%(alpha,i+1)
        response = urllib.request.urlopen(url)
        soup = BeautifulSoup(response, "lxml")
        table = soup.find("table",border=0)
        links = table.find_all("a")
        for link in links:
            artists.append({"artist":link.text, "link":link["href"]})
    return artists

def artist_songs(artist,link):
    songs = []
    response = urllib.request.urlopen(link)
    soup = BeautifulSoup(response, "lxml")
    page = soup.find("div",id="page")
    ul = page.find("ul")
    lists = ul.find_all("li")
    for l in lists:
        songs.append({"song":l.span.text, "link":l.a["href"]})
    return songs
    
def save_artist_songs(artist,link):
    songs = artist_songs(artist,link)
    artist = re.sub('\W','_',artist)
    artist = artist.replace('/','_')
    # os.mkdir("midifile/%s"%artist)
    os.makedirs("midifile/%s"%artist, exist_ok=True) # 変更
    for song in songs:
        song["song"] = re.sub('\W','_',song["song"])
        urllib.request.urlretrieve(song["link"],"midifile/%s/%s.mid"%(artist,song["song"]))

for _a in range(65,65+26):
    a =chr(_a)
    artists = alpha_artists(a)
    for artist in artists:
        save_artist_songs(artist["artist"],artist["link"])