from typing import Set

letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]


def damerau_levenshtein_one(word: str) -> Set[str]:
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]

    return set([word] + deletes + transposes + replaces + inserts)


def damerau_levenshtein(word: str, distance: int) -> Set[str]:
    if distance == 0:
        return set([word])

    edits = damerau_levenshtein_one(word)

    if distance > 0:
        edits = edits.union(dw for w in edits for dw in damerau_levenshtein(w, distance - 1))

    return edits