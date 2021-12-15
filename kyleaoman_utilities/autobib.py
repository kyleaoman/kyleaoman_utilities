import requests
import os
import re

bib_file = 'paper.bib'
aux_file = 'paper.aux'
blg_file = 'paper.blg'
api_token = None
api_service = 'https://api.adsabs.harvard.edu/v1/export/bibtex'
warn_re = 'Warning--I didn\'t find a database entry for "(.*)"\n'


def get_bibtex(bibcode):
    response = requests.get(
        '{:s}/{:s}'.format(api_service, bibcode),
        headers={'Authorization': 'Bearer {:s}'.format(api_token)}
    )
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        print('bibcode "{:s}" not found on ADS'.format(bibcode))
        raise
    else:
        print('got bibtex for bibcode "{:s}" from ADS'.format(bibcode))
        return response.text


missing_bibcodes = list()
os.system('bibtex {:s}'.format(aux_file))
with open(blg_file) as blg:
    for line in blg.readlines():
        re_result = re.search(warn_re, line)
        if re_result is not None:
            missing_bibcodes.append(re_result.group(1))

with open(bib_file, 'a+') as bib:
    for missing_bibcode in missing_bibcodes:
        try:
            missing_bibtex = get_bibtex(missing_bibcode)
        except requests.exceptions.HTTPError:
            continue
        bib.write(missing_bibtex)

os.system('bibtex {:s}'.format(aux_file))
