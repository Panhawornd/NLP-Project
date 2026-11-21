import emoji
import html
import re
import unicodedata
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

lemmatizer = WordNetLemmatizer()
tweet_tokenizer = TweetTokenizer(preserve_case=False)
keep_words = {
    "no", "nor", "not", "never", "very", "too", "so", "more", "most", "less",
    "but", "however", "yet", "up", "down",
}
stop_words = set(ENGLISH_STOP_WORDS) - keep_words

contractions = {
    "can't": "can not",
    "won't": "will not",
    "shan't": "shall not",
    "ain't": "not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "couldn't": "could not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "mustn't": "must not",
    "needn't": "need not",
    "mightn't": "might not",
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "we'll": "we will",
    "they'll": "they will",
    "let's": "let us",
}
contraction_pattern = re.compile(
    r"\b(?:" + "|".join(
        re.escape(item) for item in sorted(contractions, key=len, reverse=True)
    ) + r")\b",
    flags=re.IGNORECASE,
)

# Convert common ASCII emoticons before removing punctuation.
emoticon_variants = {
    "emoticon_broken_heart": ["</3"],
    "emoticon_heart": ["<3"],
    "emoticon_wink": [";)", ";-)", ";]", ";-]"],
    "emoticon_playful": [":p", ":-p", ";p", ";-p"],
    "emoticon_positive": [":)", ":-)", ":]", ":-]", ":d", ":-d", "=)", "=-)", "(:", "(-:"],
    "emoticon_negative": [":(", ":-(", ":[", ":-[", "=(", "=-(", ":'(", ":'-(", "):", ")-:"],
    "emoticon_neutral": [":|", ":-|", ":/", ":-/", ":\\", ":-\\", ":o", ":-o"],
}
emoticon_map = {
    variant.lower(): label
    for label, variants in emoticon_variants.items()
    for variant in variants
}
emoticon_pattern = re.compile(
    r"(?<![A-Za-z0-9])(?:" + "|".join(
        re.escape(item) for item in sorted(emoticon_map, key=len, reverse=True)
    ) + r")(?![A-Za-z0-9])",
    flags=re.IGNORECASE,
)

def clean_tweet(text):
    text = html.unescape(str(text))
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = contraction_pattern.sub(
        lambda match: contractions[match.group(0).lower()], text
    )
    text = emoticon_pattern.sub(
        lambda match: f" {emoticon_map[match.group(0).lower()]} ", text
    )
    text = emoji.demojize(text)
    # Remove HTML tags such as <br> and <b>.
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove links that start with http://, https://, or www.
    # The URL match continues until whitespace.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove @mentions made of letters, digits, or underscores.
    # (?<!\w) prevents matching an @ that appears inside another word.
    text = re.sub(r"(?<!\w)@\w+", " ", text)

    # Remove the # symbol but keep the hashtag words for classification.
    # Split hashtags at lowercase-to-uppercase boundaries,
    # so #SoHappy becomes "So Happy".
    text = re.sub(
        r"#(\w+)",
        lambda match: re.sub(r"([a-z])([A-Z])", r"\1 \2", match.group(1)),
        text,
    )

    # Preserve expressive punctuation as words before removing punctuation.
    text = re.sub(r"!+", " exclamation_mark ", text)
    text = re.sub(r"\?+", " question_mark ", text)
    text = re.sub(r"\.{2,}", " ellipsis ", text)

    # Limit any run of 3+ identical characters to two,
    # reducing stretched spellings while retaining emphasis (soooo -> soo).
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    text = text.lower()
    return "".join(
        " " if unicodedata.category(char).startswith("P") and char != "_" else char
        for char in text
    ).strip()


def wordnet_pos(penn_tag):
    """Map a Penn Treebank tag to the WordNet POS labels."""
    if penn_tag.startswith("J"):
        return "a"
    if penn_tag.startswith("V"):
        return "v"
    if penn_tag.startswith("R"):
        return "r"
    return "n"


def tokenize_normalize(text, remove_stopwords=False, lemmatize=True):
    tokens = tweet_tokenizer.tokenize(clean_tweet(text))

    if remove_stopwords:
        tokens = [
            token for token in tokens
            if not (token.isalpha() and token in stop_words)
        ]

    # POS-tag regular words and preserve emoji/emoticon placeholders.
    word_positions = [i for i, token in enumerate(tokens) if token.isalpha()]
    if lemmatize and word_positions:
        tagged_words = nltk.pos_tag([tokens[i] for i in word_positions])
        for index, (word, tag) in zip(word_positions, tagged_words):
            tokens[index] = lemmatizer.lemmatize(word, wordnet_pos(tag))

    return tokens


def identity_tokenizer(tokens):
    """Return pre-tokenized tokens unchanged for sklearn vectorizers."""
    return tokens


def join_token_lists(token_lists):
    """Join normalized tokens so a character vectorizer can read each tweet."""
    return [" ".join(tokens) for tokens in token_lists]


def average_word2vec(token_lists, embedding):
    """Average known Word2Vec vectors for each token list."""
    import numpy as np
    vectors = []
    for tokens in token_lists:
        known = [embedding.wv[token] for token in tokens if token in embedding.wv]
        vectors.append(np.mean(known, axis=0) if known else np.zeros(embedding.vector_size))
    return np.asarray(vectors, dtype=np.float32)
