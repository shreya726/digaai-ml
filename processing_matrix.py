import csv
import json
from mapping import mapping as MAPPING

MAX_NAME_LENGTH = 10
PADDING = 'Q'
with open('grams.json') as data:
	GRAMS = json.load(data)

def process(filename):
	first_names = []
	last_names = []
	sources = []
	ACCENT = False
	with open(filename, 'r') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		for row in csvreader:
			if row[0] == "b'jo\xc3\xa3ovitor'":
				continue
			try:
				x = row[0].decode('utf-8')
				y = row[1].decode('utf-8')
				ACCENT = True
			except:
				pass

			if not ACCENT:
				try:
					first = convert_to_numerical(get_3grams(row[0]))
					last = convert_to_numerical(get_3grams(row[1]))
					first_names += [first]
					last_names += [last]
					source = row[2]
					sources += [source]
				except KeyError:
					pass
	csvfile.close()

	new_filename = filename[:-4] + '_processed.csv'
	with open(new_filename, 'w') as csvfile:
		csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_NONE, delimiter=',')
		for i in range(0, len(first_names)):
			current_row = first_names[i] + last_names[i] + list(sources[i])
			csvwriter.writerow(current_row)
	csvfile.close()
	print('done')

def get_3grams(name):
	""" Get 3-grams of names 
	"""
	name = name[2:-1]
	grams = [name[i:i+3] for i in range(0, len(name) - 2)]
	if len(grams) < MAX_NAME_LENGTH:
		grams += ['QQQ'] * (MAX_NAME_LENGTH - len(grams))
	elif len(grams) > MAX_NAME_LENGTH:
		grams = grams[:10]

	return grams
	

def convert_to_numerical(grams):
	result = []
	for gram in grams:
		try:
			result += [int(GRAMS[gram])]
		except KeyError:
			converted_gram = []
			for letter in gram:
				
				if letter.encode('utf-8') == b"'"  : continue
				
				converted_gram += [(MAPPING[letter.encode('utf-8')])]
				
			num_val = '0'.join([str(c) for c in converted_gram])
			result += [int(num_val)]

			# Saving grams to dictionary
			GRAMS[gram] = num_val
	return result

process('data/training.csv')
process('data/testing.csv')
process('data/eval.csv')
