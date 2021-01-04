'''
Download PDF files for a series of law chapter numbers
found in our training CSV file provided by partners.
Then extract the chapter texts from those PDFs.
Finally insert the chapter text into a new column 'Text' of our training CSV.
Note: each PDF contains a bit of the previous chapter and following one.
'''
import difflib
import math
import os
import re
from collections import Counter

import pandas as pd
import settings
import urllib.request, urllib
import utils


DOWNLOAD_PATH = os.path.join(settings.DATA_PATH, 'in', 'download')
os.makedirs(DOWNLOAD_PATH, exist_ok=True)
TITLES_FILENAME = 'titles-1398.csv'


def download_and_extract():
    titles_df = pd.read_csv(os.path.join(settings.DATA_PATH, 'in', TITLES_FILENAME))

    # download all the pdfs
    download_pdfs(titles_df)

    # Extract the texts from PDF into .txt files
    # Using two complementary techniques:
    #   1. extract embedded text from PDF
    #   2. OCR with Tesseract
    # 2. is better than 1. in general but not always.
    for i in [0, 1]:
        extract_texts_from_pdfs(titles_df, use_tesseract=i)

    extract_chapters_from_texts(titles_df, limit=None)

    # write the new dataframe back into the input csv
    titles_df.to_csv(os.path.join(settings.DATA_PATH, 'in', TITLES_FILENAME), index=False)


def extract_chapters_from_texts(titles_df, limit=None):
    '''Extract the text of the chapters from the texts converted from the PDFs
    and insert them into a new column in titles_df dataframe called Text.'''
    found = 0
    total = len(titles_df)

    titles_df['Text'] = titles_df['Title']

    if limit is None:
        limit = total

    for i, row in titles_df.iterrows():
        if i >= limit:
            break

        chapter_number = row['Chapter']

        # print(str(chapter_number) + ' ' + '=' * 40)

        best = None
        options = []

        # We extract the desired chapter with Tesseract first.
        # If that doesn't work we fall back to the text embedded in the PDF.
        for j, prefix in enumerate(['', 'tes-']):
            filename = f"{prefix}{chapter_number}.txt"
            content = utils.read_file(os.path.join(DOWNLOAD_PATH, filename))
            chapter, chapter_first_line = extract_chapter_from_text(content, chapter_number)
            options.append({
                'filename': filename,
                'chapter': chapter,
                'first_line': chapter_first_line,
            })

            if options[-1]['chapter']:
                best = j

        if best is None:
            print(f'WARNING: chapter roman number not found {chapter_number}')
        else:
            found += 1
            text = utils.repair_ocred_text(options[best]['chapter'])

            titles_df.loc[titles_df['Chapter'] == chapter_number, "Text"] = text
            titles_df.loc[titles_df['Chapter'] == chapter_number, "First line"] = options[best]['first_line']
            # print('INFO: {} - {}'.format(chapter_number, options[best]['filename']))

    print('INFO: {} parsed. {} not found.'.format(limit, limit-found))


def extract_chapter_from_text(text, chapter_number):
    '''
    :param text: textual content of a pdf file
    :param chapter_number: number of a chapter to extract from text
    :return: a string with the content of the extracted chapter

    Method:

    All chapters in a text start with a line like this:

    CHAPTER MCDLXXXIV

    Which is the roman number of the chapter.

    We use some text similarity function to find all the lines with
    a chapter number (chapters). Then we look for perfect match on the
    number we are after of the following one.

    We replace some characters and patterns which are often misencoded
    or badly OCRed in the pdf to improve the chances of a match.
    '''
    ret = ''

    roman = utils.get_roman_from_int(chapter_number)
    '''
    e.g. 1484
    ['CHAPTER MODLXXXIV', 'CHAPTER MCDLXXXIIL', 'CHAPTER MCDLXXXY¥']
    
    e.g. 1485
    # First match is 1486, second is what we want for 1485.
    ['CHAPTER MCDLXXXVI', 'CHAPTER MCDLXXXY¥']
    '''
    # clean up to facilitate matches of chapter number
    # Tesseract often reads O when it is written C
    text_normalised = re.sub(r'[^A-Z1-9\n]+', r' ', text.upper().replace('1', 'I'))
    text_normalised = utils.normalise_roman_number(text_normalised)
    lines = [line.strip('. ') for line in text_normalised.split('\n')]

    marker = utils.normalise_roman_number(f'CHAPTER {roman}')
    # list of lines with CHAPTER XXX
    chapters = [
        c for c in difflib.get_close_matches(marker, lines)
        # exclude footnotes, e.g. *CHAPTER 1478
        if not(re.search(r'\d', c) and len(c) < 16)
    ]
    # sort them by appearance rather than similarity.
    # Similarity is not reliable due to corruption of the numbers by OCR.
    chapters = [line for line in lines if line in chapters]

    start = None
    end = None
    warning = 'NOT FOUND'
    if len(chapters) == 1:
        # only one match, we assume it's what we are after even if not perfect
        start = lines.index(chapters[0])
        end = len(lines)
    elif len(chapters) > 1:
        # return a line which ends with the same roman number
        start = find_exact_chapter_number(chapter_number, chapters, lines)
        # line for next chapter number
        end = find_exact_chapter_number(chapter_number + 1, chapters, lines)
        if start is not None and end is not None and not(chapters.index(lines[end]) == chapters.index(lines[start]) + 1):
            warning = 'NEXT != THIS + 1'
            start = None
            end = None
        if start is None:
            if end is not None:
                start = max([
                    lines.index(ch)
                    for ch in chapters
                    if lines.index(ch) < end
                ] + [
                    -1
                ])
                if start == -1:
                    warning = 'NEXT is first'
                    start = None

    if start is None:
        # heuristic: if no good match AND we have two candidate chapters
        # then pick the first candidate.
        if len(chapters) == 2:
            print(chapters)
            start = lines.index(chapters[0])
            end = lines.index(chapters[1], start+1)

    # now get all the lines between start and end
    first_line = ''
    if start is not None:
        if end is None:
            end = min([
                lines.index(ch)
                for ch in chapters
                if lines.index(ch) > start
            ] + [
                len(lines)
            ])
            if end != len(lines) and lines[end]:
                if re.findall(r'\d', lines[end]):
                    warning = 'END might be a footnote'
                    print(warning, lines[end])
        if re.findall(r'\d', lines[start]):
            warning = 'START might be a footnote'
            print(warning, lines[start])

        # extract lines [start:end] from the non-normalised text
        lines = text.split('\n')
        ret = '\n'.join(lines[start+1:end]).strip('\n ')
        first_line = lines[start].strip('\n ')

    if not ret:
        print(chapter_number, repr(marker), chapters, warning)

    return ret, first_line


def find_exact_chapter_number(number, candidates_lines, all_lines):
    '''return index of the line with an exact match for given number'''
    ret = None
    roman = utils.get_roman_from_int(number)
    roman_normalised = utils.normalise_roman_number(roman)
    exact_chapters = [
        ch for ch in candidates_lines
        if ch.endswith(roman_normalised)
    ]
    if len(exact_chapters) == 1:
        ret = all_lines.index(exact_chapters[0])

    return ret


def download_pdfs(titles_df):
    '''
    Download the PDFs from palrb.us website.
    Skip files already on disk.
    '''
    total = len(titles_df)

    for i, row in titles_df.iterrows():
        yd = str(row['Session'])[1]
        url = f"http://www.palrb.us/statutesatlarge/1{yd}001{yd}99/{row['Session']}/0/act/{row['Chapter']}.pdf"

        out_path = os.path.join(DOWNLOAD_PATH, f"{row['Chapter']}.pdf")

        if utils.download(url, out_path) == 2:
            print(f"{i}/{total} {url}")


def extract_texts_from_pdfs(titles_df, reprocess=False, read_only=False, limit=None, use_tesseract=False):
    found = 0
    total = len(titles_df)

    if limit is None:
        limit = total
    prefix = ''
    if use_tesseract:
        prefix = 'tes-'

    for i, row in titles_df.iterrows():
        if i >= limit:
            break

        pdf_path = os.path.join(DOWNLOAD_PATH, f"{row['Chapter']}.pdf")
        txt_path = os.path.join(DOWNLOAD_PATH, f"{prefix}{row['Chapter']}.txt")

        if not os.path.exists(pdf_path):
            print('WARNING: {pdf_path} not found.')

        if reprocess or not os.path.exists(txt_path):
            print(f"{i}/{total} {pdf_path}")
            content = utils.extract_text_from_pdf(
                pdf_path,
                use_tesseract=use_tesseract
            )

            if content and not read_only:
                with open(txt_path, 'wt') as fh:
                    fh.write(content)

    warnings_count = limit


def get_first_chapter_from_pdf_text(text, print_chapters=False, chapter_number=None):
    ret = text

    '''
    Passed December 9, 1789.
    Recorded L. B. No. 4, p. 56.
    '''
    # chapters = re.split(r'(?ism)(?:passed|approved) .{,30}recorded.*?$', text)

    # chapters = re.split(r'(?sm)^.{,2}APTER (M[^\n]{,12})$', text)
    if chapter_number:
        roman = utils.get_roman_from_int(chapter_number)
        # chapters = re.split(r'(?sm)^.{,2}APTER (M[^\n]{,12})$', text)

        pattern = r'(?sm)\s{}\.'.format(re.escape(roman))

        print(pattern)

        chapters = re.findall(pattern, text)

    if print_chapters:
        for i, c in enumerate(chapters):
            print(str(i) + '-'*40)
            print(c)

    if len(chapters) < 1:
        ret = None
    else:
        ret = text

    return ret


def check_extraction_quality(warnings_to_show=None):
    titles_df = pd.read_csv(os.path.join(settings.DATA_PATH, 'in', TITLES_FILENAME))

    def find_warnings(text, chapter_number, first_line):
        '''returns a list of warning flags'''
        ret = []

        if pd.isna(first_line):
            first_line = ''

        if not re.search(r'(?i)^\W*\w{2,3}apter\b', first_line):
            ret.append('NO_CHAPTER')

        if len(text) < 400:
            ret.append('SHORT_TEXT')

        if utils.normalise_roman_number(utils.get_roman_from_int(chapter_number)) not in utils.normalise_roman_number(first_line):
            ret.append('NO_CHAPTER_NUMBER')

        return ret

    stats = Counter({'TOTAL': len(titles_df)})

    for idx, row in titles_df.iterrows():
        warnings = find_warnings(row['Text'], row['Chapter'], row['First line'])
        stats.update(warnings)
        if (warnings and warnings_to_show is None) or (set(warnings).intersection(set(warnings_to_show))):
            print(
                '{} {:20.20} {} {:40.40} {:80.80} {}'.format(
                    row['Chapter'],
                    row['First line'],
                    str(row['Class 3']).zfill(3),
                    row['Title'],
                    repr(re.sub(r'\s+', ' ', row['Text'])),
                    ' '.join(warnings)
                )
            )

    print('\nWarnings:')
    print(stats)


download_and_extract()

check_extraction_quality(['NO_CHAPTER'])
