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

key = "INSERT ACCESS TOKEN"

def get_names(filename, group_ids, limit):
    f = open(filename, "w")
    f.seek(0)
    writer = csv.writer(f, quoting=csv.QUOTE_NONE)
    count = 0
    
    for group_id in group_ids:
        conn = http.client.HTTPSConnection("graph.facebook.com")
        next_link = " "
        while not next_link == None:
            names = []
            conn.request("GET", "/v2.10/" + group_id + "/members?access_token=" + key + "&fields=name&after=" + next_link)
            res = conn.getresponse()
            data = res.read()
            json_data = json.loads(data)
            
            if "data" not in json_data:
                print(json_data["error"]["message"])
                return

            for user in json_data["data"]:
                data_name = user["name"].lower()
                pre_name = data_name.split()
                fname = ' '.join(pre_name[:-1])
                lname = pre_name[-1]
                name = [fname, lname]

                writer.writerow(name)
                count += 1
                if count >= limit:
                    f.close()
                    return
            

            if "paging" in json_data:
                next_link = str(json_data["paging"]["cursors"]["after"])
            else:
                next_link = None
    
            
    f.close()
    return

# These are the group ids of the groups we are taking names from.
brazil_ids = ["422121401312686","701457699900185","435932449946192","538824349628836","248974475276092"]
portugal_ids = ["196549537052388","152035978166262","795722390452796","1748734022116488","344851139179868","481561982041744"]

# Default limit is one million
limit=1000000

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--brazil", help="Gather only data for Brazilians", action="store_true")
parser.add_argument("-p", "--portugal", help="Gather only data for Portuguese", action="store_true")
parser.add_argument("-l", "--limit", help="Specify a limit for how many names to gather")
args = parser.parse_args()

if args.limit:
    limit = int(args.limit)

if not args.brazil and not args.portugal:
    get_names("brazilians.csv", brazil_ids, limit)
    get_names("portuguese.csv", portugal_ids, limit)

if args.brazil:
    get_names("brazilians.csv", brazil_ids, limit)
elif args.portugal:
    get_names("portuguese.csv", portugal_ids, limit)
