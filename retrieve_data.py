key = "SECRET"

import http.client
import json
import csv

def get_names(filename, group_ids):
    f = open(filename, "w")
    writer = csv.writer(f, quoting=csv.QUOTE_NONE)
    
    for group_id in group_ids:
        conn = http.client.HTTPSConnection("graph.facebook.com")
        next_link = " "
        while not next_link == None:
            names = []
            conn.request("GET", "/v2.10/" + group_id + "/members?access_token=" + key + "&fields=name&after=" + next_link)
            res = conn.getresponse()
            data = res.read()
            json_data = json.loads(data)
        
            for user in json_data["data"]:
                name = [user["name"]]
                writer.writerow(name)

            if "paging" in json_data:
                next_link = str(json_data["paging"]["cursors"]["after"])
            else:
                next_link = None
    
    f.close()

brazil_ids = ["422121401312686"]
portugal_ids = []

get_names("brazilians.csv", brazil_ids)
get_names("portuguese.csv", portugal_ids)
