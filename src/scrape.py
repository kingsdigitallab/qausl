'''
Not found:
'''
import difflib
import os
import re
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

        # We extract the desired chapter from Tesseract first.
        # If that doesn't work we fall back to the text embedded in the PDF.
        for j, prefix in enumerate(['', 'tes-']):
            filename = f"{prefix}{chapter_number}.txt"
            content = utils.read_file(os.path.join(DOWNLOAD_PATH, filename))
            options.append({
                'filename': filename,
                'chapter': extract_chapter_from_text(content, chapter_number),
            })

            if options[-1]['chapter']:
                best = j

        if best is None:
            print(f'WARNING: chapter not found {chapter_number}')
        else:
            found += 1
            titles_df.loc[titles_df['Chapter'] == chapter_number, "Text"] = options[best]['chapter']
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
    roman = roman.replace('L', 'I')
    text_cleaned = re.sub(r'[^A-Z\n]+', r' ', text.upper().replace('1', 'I'))
    text_cleaned = text_cleaned.replace('O', 'C').replace('L', 'I').replace('Y', 'V')
    # replace nonsensical I/L, e.g. 1492: MCDLXCII => MCDXCII
    text_cleaned = text_cleaned.replace('IXC', 'XC')
    lines = [line.strip('. ') for line in text_cleaned.split('\n')]

    marker = f'CHAPTER {roman}'
    # list of lines with CHAPTER XXX
    chapters = difflib.get_close_matches(marker, lines)

    start = None
    end = None
    show = None
    if len(chapters) > 1:
        # return a line which ends with the same roman number
        exact_chapters = [ch for ch in chapters if ch.endswith(roman)]
        if len(exact_chapters) == 1:
            start = lines.index(exact_chapters[0])
        else:
            # search failed, we then look for next chapter number
            roman_next = utils.get_roman_from_int(chapter_number + 1)
            roman_next = roman_next.replace('L', 'I')
            exact_chapters = [ch for ch in chapters if
                              ch.endswith(roman_next)]
            if len(exact_chapters) == 1:
                end = lines.index(exact_chapters[0])
                start = max([
                    lines.index(ch)
                    for ch in chapters
                    if lines.index(ch) < end
                ] + [
                    0
                ])
                show = 0
    elif len(chapters) == 1:
        # only one match, we assume it's what we are after even if not perfect
        start = lines.index(chapters[0])

    # now get all the lines between start and end
    if start and not end:
        end = min([
            lines.index(ch)
            for ch in chapters
            if lines.index(ch) > start
        ] + [
            len(lines)
        ])

    if start is not None:
        lines = text.split('\n')
        ret = '\n'.join(lines[start:end])
        if show:
            print(chapter_number, repr(marker), chapters)
            print(ret)

    if not ret:
        print(chapter_number, repr(marker), chapters)

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

        if not os.path.exists(out_path):
            print(f"{i}/{total} {url}")
            try:
                with urllib.request.urlopen(url) as resp:
                    with open(out_path, 'wb') as fh:
                        fh.write(resp.read())
            except urllib.error.HTTPError as e:
                print(f"ERROR: {url} {e}")


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


download_and_extract()
