from json import loads
from typing import List, Tuple

CONFIDENCE_PATH: str = "../confidence.json"

def load_confidence(path: str) -> List[List[float]]:
    confidence: List[List[float]] = []
    with open(path, 'r') as file:
        confidence = loads(file.read())

    return confidence

def main() -> None:
    confidence: List[List[float]] = load_confidence(CONFIDENCE_PATH)
    min0, max1 = min(confidence, key=lambda e: e[0])
    min1, max0 = min(confidence, key=lambda e: e[1])

    in_range: List[Tuple[int, List[float]]] = []
    for i, (c1, c2) in enumerate(confidence):
        if 0.4975 <= c1 < 0.5025:
            in_range.append((i, (c1, c2)))


    print(min0, max1)
    print(min1, max0)
    print(len(in_range))
    print(in_range)

    return

if __name__ == "__main__":
    main()