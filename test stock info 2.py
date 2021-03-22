#import urllib
import re
import sys
import urllib.request as ur
def get_quote(symbol):
    base_url = 'http://finance.google.com/finance?q='

    
    #content = urllib.urlopen(base_url + symbol).read()
    with ur.urlopen(base_url + symbol) as url:
        content = url.read()
    
    m = re.search('id="ref_694653_l".*?>(.*?)<', content)
    if m:
        quote = m.group(1)
    else:
        quote = 'no quote available for: ' + symbol
    return quote
print(get_quote('goog'))
