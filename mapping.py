# Mapping of letters

alphabets = alphabets = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

accents = ["á", "â", "ã", "à", "ç", "é", "ê", "í", "ó", "ô", "õ", "ú"]

for i in range(len(accents)):
    accents[i] = accents[i].encode('utf-8')

mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 11, 'k': 12, 'l': 13, 'm': 14, 'n': 15, 'o': 16, 'p': 17, 'q': 18, 'r': 19, 's': 21, 't': 22, 'u': 23, 'v': 24, 'w': 25, 'x': 26, 'y': 27, 'z': 28, b'\xc3\xa1': 29, b'\xc3\xa2': 31, b'\xc3\xa3': 32, b'\xc3\xa0': 33, b'\xc3\xa7': 34, b'\xc3\xa9': 35, b'\xc3\xaa': 36, b'\xc3\xad': 37, b'\xc3\xb3': 38, b'\xc3\xb4': 39, b'\xc3\xb5': 41, b'\xc3\xba': 42}

