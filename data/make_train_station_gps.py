import urllib
import urllib2
import json
import bs4

# Please do not execute this script too many times because Google limits
# the amount of GeoData queries that can be executed. As a safeguard I have
# commented the line all the way at the bottom of the script that defines
# the API key so you'll need to uncomment that as a confirmation that you've
# read this.

def get_gps(name, one=False, tld="nl"):
    name = name.encode('utf-8')
    j = json.load(urllib2.urlopen("https://maps.googleapis.com/maps/api/geocode/json?%s" % urllib.urlencode({"key": key, "address": name, "region": tld})))
    if one:
        if j["results"]:
            obj = j["results"][0]
            return obj["geometry"]["location"]["lng"], obj["geometry"]["location"]["lat"]
        else:
            return None
    else:
        return [(obj["geometry"]["location"]["lng"], obj["geometry"]["location"]["lat"]) for obj in j["results"]]

def explore_tag(tag):
    watching = False
    results = []
    
    for child in tag.children:
        if type(child) is bs4.Tag:
            if child.name == 'h2':
                watching = child.span.text in letters
            elif child.name == 'div' and child.ul and watching:
                results.extend(explore_ul(child.ul))
            elif child.name == 'ul' and watching:
                results.extend(explore_ul(child))

    return results

def explore_ul(tag):
    results = []

    for li in tag.find_all('li', recursive=False):
        if li.find('a', recursive=False):
            results.append(li.a.get('title'))
            
        if li.ul:
            results.extend(explore_ul(li.ul))

    return results

if __name__ == "__main__":
    # Scrape names from wikipedia
    wiki_url = "http://en.wikipedia.org/wiki/List_of_railway_stations_in_Luxembourg"
    soup = bs4.BeautifulSoup(urllib2.urlopen(wiki_url))
    letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ["IJ"]

    cc = soup.find('div', {'id': 'mw-content-text'})

    stations = explore_tag(cc)
    if stations:
        print "Found %d stations, from %s to %s." % (len(stations), stations[0], stations[-1])
    else:
        print "No stations found."

    # Find GPS coordinates from google maps
    key = "AIzaSyB8D7a8gpUYJf62tgIrw5R2RTCNapR1WF4"
    gps = {s: get_gps(s, True, "lu") for s in stations}

    gps_found = {k: v for k, v in gps.iteritems() if v is not None}
    no_gps_found = [k for k, v in gps.iteritems() if v is None]

    print "GPS coordinates not found for %d stations" % len(no_gps_found)
