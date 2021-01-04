'''
Download the titles from QUB's Irish Legislation Database.
Save the data in a csv with the following columns:
    index: sequential, starts at 10000
    Session: the year (it might be suffixed with a letter, e.g. 1796B)
    Short Title: a descriptive sub-category below level 3
    Title: the title of the chapter
    Class 3: 3-digit code of the legal topic category (compatible with QAUSL)
    1st alt Class 3: empty, only there to keep cvs compatible with QAUSL input

'''
import os
import pandas as pd
import settings
import utils
import re

DOWNLOAD_PATH = os.path.join(settings.DATA_PATH, 'in', 'qub')
os.makedirs(DOWNLOAD_PATH, exist_ok=True)
TITLES_FILENAME = 'titles-1398.csv'

CSV_OUT_PATH = os.path.join(settings.DATA_PATH, 'in', 'titles-qub.csv')

def download_and_extract():
    df_titles_qausl = pd.read_csv(os.path.join(settings.DATA_PATH, 'in', TITLES_FILENAME))

    get_all_categories = True

    if get_all_categories:
        # all possible categories according to our topic scheme of reference
        categories = utils.read_class_titles(3)
    else:
        # only categories used in QAULS training set
        categories = df_titles_qausl['Class 3'].apply(
            lambda c: re.sub(r'\..*', r'', str(c))[:3].zfill(3)
        ).unique()

    total = len(categories)

    # download all the htmls
    for i, category in enumerate(categories):
        url = f'https://www.qub.ac.uk/ild/?func=advanced_search&search=true&search_string=&search_string_type=ALL&search_type=any&session_from=1692&session_to=1800&enacted_search=all&subjects=%7C{category}%7C'
        out_path = os.path.join(DOWNLOAD_PATH, f'{category}.html')
        if utils.download(url, out_path) == 2:
            print(f'{i+1}/{total} [{category}] downloaded {url}')

    # extract the content
    df_titles = pd.DataFrame()
    columns = {0: 'Session', 1: 'Short Title', 2: 'Title'}
    for i, category in enumerate(categories):
        print(f'{i + 1}/{total} {category} extracting')

        warning = ''
        out_path = os.path.join(DOWNLOAD_PATH, f'{category}.html')
        content = utils.read_file(out_path)
        dfs = pd.read_html(content)
        if len(dfs) > 2:
            df = dfs[2]
            if 'Matching Record' in df[2][0]:
                df = df[2:].rename(columns=columns)[columns.values()]
                df['Class 3'] = category
                df['1st alt Class 3'] = ''
                df_titles = df_titles.append(df, ignore_index=True)
            else:
                warning = 'titles not found'
        else:
            warning = 'no results'

        if warning:
            print(f'WARNING: [{category}] {warning}')

    df_titles['Chapter'] = df_titles.index + 10000

    df_titles.to_csv(CSV_OUT_PATH, index=False, columns=[
        'Chapter', 'Session', 'Class 3', '1st alt Class 3', 'Title', 'Short Title',
    ])
    print(f'Done. {len(df_titles)} titles saved in {CSV_OUT_PATH}')

download_and_extract()
