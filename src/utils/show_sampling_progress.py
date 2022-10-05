
from typing import List
list_of_colors_from_red_to_blue = [f"\033[38;2;{r};0;{b}m" for r, b in zip(range(255, 0, -10), range(0, 255, 10))]

def pprint_sentences(sentences: List[str], banner: str = "", sep: str = ""):
    """
    Given a list of sentences, prints them with a gradient of colors from red to blue
    """
    print()
    print(f"\033[1m{'=' * 20} {banner} {'=' * 20}\033[0m")
    for i, sentence in enumerate(sentences):
        sentence_color = list_of_colors_from_red_to_blue[i]
        if i == len(sentences) - 1:
            print(f"\033[38;5;{sentence_color}{sentence}\033[0m")
        else:
            print(f"\033[38;5;{sentence_color}{sentence}\033[0m", end=sep)
    print()
    

if __name__ == '__main__':
    sentences = [
        "This is a sentence",
        "This is another sentence",
        "This is a third sentence",
        "This is a fourth sentence",
        "This is a fifth sentence",
        "This is a sixth sentence",
        "This is a seventh sentence",
        "This is an eighth sentence",
        "This is a ninth sentence",
        "This is a tenth sentence",
        "This is an eleventh sentence",
        "This is a twelfth sentence",
        "This is a thirteenth sentence",
        "This is a fourteenth sentence",
        "This is a fifteenth sentence",
        "This is a sixteenth sentence",
        "This is a seventeenth sentence",
        "This is an eighteenth sentence",
        "This is a nineteenth sentence",
        "This is a twentieth sentence",
    ]
    for i in range(1, len(sentences) + 1):
        pprint_sentences(sentences[:i], sep= " -> ")
        print("---")

