from bs4 import BeautifulSoup
import requests
import re
for x in range(0,50,10):
 def down_midi(url,filename):
    r = requests.get(url)
    print("down OK:"+filename)
    with open(filename,'wb') as f:
        f.write(r.content)
    f.close()

 r = requests.get('http://abcnotation.com/searchTunes?q=Japanese&f=c&o=a&s='+str(x)).content.decode('UTF-8')
 soup = BeautifulSoup(r,'lxml')
 html = soup.prettify()

 midisource = soup.find_all('object',type="audio/mid")
 for link in midisource:
    links = re.findall('data="([^"]+)"',str(link))
    filename = re.findall('media/(.*)\?a=',str(links))
    filename = str(filename)[2:-2]
    links = str(links)[2:-2]
    print(links)
    print(filename)
    down_midi(links,filename)
