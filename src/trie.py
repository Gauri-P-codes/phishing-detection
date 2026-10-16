"""Self-study: Data-structure management -> Trie for fast keyword / brand lookup.
Insert O(L); substring scan O(len(text) * L_max) independent of how many words are stored."""
class Trie:
    def __init__(self): self.root = {}
    def insert(self, word):
        node = self.root
        for ch in word: node = node.setdefault(ch, {})
        node["$"] = True
    def contains(self, word):
        node = self.root
        for ch in word:
            if ch not in node: return False
            node = node[ch]
        return "$" in node
    def find_in_text(self, text):
        """True if ANY stored word occurs as a substring of text."""
        for i in range(len(text)):
            node = self.root
            for ch in text[i:]:
                if ch not in node: break
                node = node[ch]
                if "$" in node: return True
        return False

SUSPICIOUS = ["login","secure","verify","update","account","confirm","banking","signin","webscr","password"]
BRANDS = ["google","paypal","amazon","microsoft","apple","netflix","sbi","hdfcbank","icicibank","facebook","paytm"]
KEYWORD_TRIE, BRAND_TRIE = Trie(), Trie()
for w in SUSPICIOUS: KEYWORD_TRIE.insert(w)
for w in BRANDS: BRAND_TRIE.insert(w)
