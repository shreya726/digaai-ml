'''
This module retrieves the names of people in Facebook groups that
relate to Brazil and Portugal using the Facebook Graph Explorer API.

The names are written to a CSV file where the first column is first names and
middle names, and the second column is last names.

To get an access token, go to https://developers.facebook.com/tools/explorer

Used for a Machine Learning project to distinguish Brazilian and Portuguese names.
'''

import http.client
import json
import csv
import argparse
import regex

key = "EAACEdEose0cBALsqxTaHjAVNZCZCt3aixAuJKCTdG1yG4eOLSH1ojtQRGUObeXU7MIEuT23PlwbAFbhliZBrWU1ULwzMqPrbGY4qMnnVDrksWxVpihZB5PyCrbIMDxXTDXKCUhF3WiGrthoKUOdMATPRgOwk8qNo77VgYyhcmcZCmYqH9JJhhUAvVVBVfD4UZD"

def get_names(group_ids, limit, value):
	f_training = open("training.csv", "a")
	writer_training = csv.writer(f_training, quoting=csv.QUOTE_NONE)

	f_testing = open("testing.csv", "a")
	writer_testing = csv.writer(f_testing, quoting=csv.QUOTE_NONE)

	f_eval = open("eval.csv", "a")
	writer_eval = csv.writer(f_eval, quoting=csv.QUOTE_NONE)
	count = 0
	
	for group_id in group_ids:
		conn = http.client.HTTPSConnection("graph.facebook.com")
		next_link = " "
		while not next_link == None:
			names = []
			conn.request("GET", "/v2.10/" + group_id + "/members?access_token=" + key + "&fields=name&after=" + next_link)
			res = conn.getresponse()
			data = res.read()
			try:
				json_data = json.loads(data)
			except:
				print("JSON data failed to load. Did you add the access token?")
				return

			if "data" not in json_data:
				print(json_data["error"]["message"])
				return

			for user in json_data["data"]:
				data_name = user["name"].lower()
				pre_name = data_name.split()
				fname = ' '.join(pre_name[:-1])
				fname = fname.replace(","," ")
				lname = pre_name[-1]
				lname = lname.replace(","," ")
				
				fname = regex.sub(u'[^\p{Latin}]', u'', fname)
				lname = regex.sub(u'[^\p{Latin}]', u'', lname)
				if fname == "" or lname == "":
					continue
				
				fname = fname.encode("utf-8")
				lname = lname.encode("utf-8")
				
				name = [fname, lname, value]
				
				if count % 5 == 0 and count % 2 == 0:
					writer_testing.writerow(name)
				elif count % 5 == 0:
					writer_eval.writerow(name)
				else:
					writer_training.writerow(name)
				
				count += 1
				if count >= limit:
					f_training.close()
					f_testing.close()
					f_eval.close()
					return
			

			if "paging" in json_data:
				next_link = str(json_data["paging"]["cursors"]["after"])
			else:
				next_link = None
	
			
	f_training.close()
	f_testing.close()
	f_eval.close()
	return

# These are the group ids of the groups we are taking names from.
brazil_ids = ["422121401312686","435932449946192","538824349628836","248974475276092"]
not_brazil_ids = ["369769286554402", "135263893484028", "716828768412499"]
# Default limit is one million
limit=1000000

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--limit", help="Specify a limit for how many names to gather")
args = parser.parse_args()

if args.limit:
	limit = int(args.limit)

get_names(brazil_ids, limit, 1)
get_names(not_brazil_ids, limit, 0)
