import unicodedata


def strip_accents_and_lowercase(s):
    # ? Fuction that removes punctuation (needed in greek) and lowers text
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn').lower()
