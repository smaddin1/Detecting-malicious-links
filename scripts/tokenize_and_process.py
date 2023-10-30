from argparse import ArgumentParser, Namespace
from json import dumps, loads
from pandas import read_csv
from sys import exit, stdout
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

TRAIN_PATH_IN: str = "../data_raw/Webpages_Classification_train_data.csv"
TEST_PATH_IN: str = "../data_raw/Webpages_Classification_test_data.csv"
TRAIN_PATH_OUT: str = "../data_processed/train.ldjson"
TEST_PATH_OUT: str = "../data_processed/test.ldjson"
LOC_TOKS_PATH: str = "../data_processed/loctoks.json"
TLD_TOKS_PATH: str = "../data_processed/tldtoks.json"
CON_TOKS_PATH: str = "../data_processed/contoks.json"
USED_FEATURES: List[str] = ["url", "ip_add", "geo_loc", "url_len", "tld", "who_is", "https", "label", "content"]

def load_data(path: str) -> Any:
    raw_data: Any = read_csv(path)
    return raw_data[USED_FEATURES]

def write_ldjson(objs: List[Dict[Any, Any]], out_path: str) -> None:
    with open(out_path, 'w') as file:
        for obj in objs:
            file.write(dumps(obj) + '\n')

    return

# url: str -> List[int] // (ex. ???)
# geo_loc: str -> int // (ex. {China, United States} -> {0, 1})
# tld: str -> int // (ex. {com, net} -> {0, 1})
def tokenize_features(test_frame: Any, train_frame: Any) -> Tuple[List[str], List[str]]:
    loc: List[str] = []
    tld: List[str] = []
    con: List[str] = []

    for frame in [test_frame, train_frame]:
        for i, row in frame.iterrows():
            """if row["geo_loc"] not in loc:
                loc.append(row["geo_loc"])
            if row["tld"] not in tld:
                tld.append(row["tld"])"""
            if row["content"] not in con and frame is not train_frame:
                con.append(row["content"])

            if i % 10000 == 0:
                print(f"Tokenization: {i} done")
                stdout.flush()

        print("Done iterating over frame!")
        with open(CON_TOKS_PATH, 'w') as file:
            file.write(dumps(con))

        stdout.flush()

    return loc, tld, con

def write_toks(loc_toks: List[str], tld_toks: List[str], con_toks: List[str]) -> None:
    with open(LOC_TOKS_PATH, 'w') as file:
        file.write(dumps(loc_toks))

    with open(TLD_TOKS_PATH, 'w') as file:
        file.write(dumps(tld_toks))

    with open(CON_TOKS_PATH, 'w') as file:
        file.write(dumps(con_toks))

    return

def load_toks() -> Tuple[List[str], List[str], List[str]]:
    loc_toks: List[str] = None
    tld_toks: Dict[Any, Any] = None
    
    with open(LOC_TOKS_PATH, 'r') as file:
        loc_toks = loads(file.read())

    with open(TLD_TOKS_PATH, 'r') as file:
        tld_toks = loads(file.read())

    with open(CON_TOKS_PATH, 'r') as file:
        con_toks = loads(file.read())

    return loc_toks, tld_toks, con_toks

def process_row(row: Any, loc_toks: List[str], tld_toks: List[str], con_toks: List[str]) -> Dict[str, Any]:
    obj: Dict[str, Any] = {}

    parsed_url = urlparse(row["url"])
    #obj["ip_add"] = list(map(lambda e: int(e), row["ip_add"].strip().split('.')))
    obj["geo_loc"] = loc_toks.index(row["geo_loc"])  # maybe toss
    obj["url_len"] = row["url_len"]
    obj["netloc_len"] = len(parsed_url.netloc)
    obj["path_len"] = len(parsed_url.path)
    obj["param_len"] = len(parsed_url.params)  # maybe toss
    obj["query_len"] = len(parsed_url.query)
    obj["frag_len"] = len(parsed_url.fragment)  # maybe toss
    obj["tld"] = tld_toks.index(row["tld"])
    obj["who_is"] = 0 if row["who_is"] == "incomplete" else 1
    obj["https"] = 0 if row["https"] == "no" else 1
    obj["content"] = con_toks[row["content"]]
    obj["label"] = 0 if row["label"] == "bad" else 1

    return obj

def load_ldjson(path: str) -> List[Dict[Any, Any]]:
    loaded: List[Dict[Any, Any]] = []

    with open(path, 'r') as file:
        line: str = file.readline()
        while line:
            loaded.append(loads(line))
            line = file.readline()

    return loaded

def normalize(test_objs: List[Dict[Any, Any]], train_objs: List[Dict[Any, Any]]) -> Tuple[List[Dict[Any, Any]], Dict[Any, Any]]:
    max_values = {'geo_loc': 233, 'url_len': 721, 'netloc_len': 71, 'path_len': 594, 'param_len': 138, 'query_len': 655, 'frag_len': 140, 'tld': 1351, 'who_is': 1, 'https': 1, 'label': 1}
    normalized_test = [{key: ((value/max_values[key]) if key != USED_FEATURES[-1] else value) for key, value in obj.items()} for i, obj in enumerate(test_objs)]
    normalized_train = [{key: ((value/max_values[key]) if key != USED_FEATURES[-1] else value) for key, value in obj.items()} for i, obj in enumerate(train_objs)]

    return normalized_test, normalized_train

def main() -> None:
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("-t", action="store_true", default=False, help="Fresh tokenization of features or load them?")
    parser.add_argument("-n", action="store_true", default=False, help="Normalizes .ldjson files.")
    args: Namespace = parser.parse_args()

    if args.n:
        print("Loading data.")
        stdout.flush()
        test_frame: Any = load_data(TEST_PATH_IN)
        train_frame: Any = load_data(TRAIN_PATH_IN)
        print("Loaded frames.")
        stdout.flush()

    if args.t:
        print("About to tokenize.")
        stdout.flush()
        loc_toks, tld_toks, con_toks = tokenize_features(test_frame, train_frame)
        write_toks(loc_toks, tld_toks, con_toks)
        print("Generated and wrote tokenization files.")
        stdout.flush()
    else:
        loc_toks, tld_toks, con_toks = load_toks()

    if args.n:
        for frame, path in [(test_frame, TEST_PATH_OUT), (train_frame, TRAIN_PATH_OUT)]:
            processed_objs: List[Dict[Any, Any]] = []
            for i, row in frame.iterrows():
                row_processed: Any = process_row(row, loc_toks, tld_toks, con_toks)
                if row_processed is not None:
                    processed_objs.append(row_processed)
                else:
                    print(f"Ignored idx {i}")

                if i % 10000 == 0:
                    print(f"Processed {i} rows.")
            print("Processed all rows in frame.")

            write_ldjson(processed_objs, path)
            print(f"Wrote processed dataset to {path}")

        return
    else:
        test_objs: List[Dict[Any, Any]] = load_ldjson(TEST_PATH_OUT)
        train_objs: List[Dict[Any, Any]] = load_ldjson(TRAIN_PATH_OUT)

        normalized_test, normalized_train = normalize(test_objs, train_objs)

        write_ldjson(normalized_test, TEST_PATH_OUT)
        write_ldjson(normalized_train, TRAIN_PATH_OUT)

if __name__ == "__main__":
    main()