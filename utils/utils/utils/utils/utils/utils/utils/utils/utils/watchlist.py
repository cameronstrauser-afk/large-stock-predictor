import json
FILE="watchlist.json"

def load_watchlist():
    try:
        with open(FILE,"r") as f:
            return json.load(f)
    except:
        return []

def save_watchlist(w):
    with open(FILE,"w") as f:
        json.dump(w,f)
