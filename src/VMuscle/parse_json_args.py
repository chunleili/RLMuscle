
def parse_json_args(args,json_path=""):
    # if not args.use_json:
    #     return
    import json
    import os
    if not os.path.exists(json_path):
        assert False, f"json file {json_path} not exist!"
    print(f"CAUTION: using json config file {json_path} to overwrite the command line args!")
    if json_path=="" and os.path.exists("config"):
        print("Using config file  to set json path")
        with open("config") as f:
            json_path = f.read().strip()
            args.json_path = json_path
    else:
        Warning("No json")
    print(f"use json_path: {json_path}")
    with open(json_path, "r") as json_file:
        config = json.load(json_file)
    for key, value in config.items():
        if hasattr(args,key):
            if getattr(args,key) != value:
                # print(f"overwriting {key} from {getattr(args,key)} to {value}")
                setattr(args,key,value)
        else:
            # print(f"Add new json key {key}:{value} to args")
            setattr(args,key,value)
    return args