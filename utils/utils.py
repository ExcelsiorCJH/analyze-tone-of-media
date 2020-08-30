import platform

from konlpy.tag import Okt, Komoran, Hannanum, Kkma

if platform.system() == "Windows":
    try:
        from eunjeon import Mecab
    except:
        print("please install eunjeon module")
else:  # Ubuntu일 경우
    from konlpy.tag import Mecab

from utils.types_ import *


def get_tokenizer(tokenizer_name: str = "mecab") -> "tokenizer":
    if tokenizer_name == "komoran":
        tokenizer = Komoran()
    elif tokenizer_name == "okt":
        tokenizer = Okt()
    elif tokenizer_name == "mecab":
        tokenizer = Mecab()
    elif tokenizer_name == "hannanum":
        tokenizer = Hannanum()
    elif tokenizer_name == "kkma":
        tokenizer = Kkma()
    else:
        return None

    return tokenizer


def get_tokens(
    data: List[Tuple], save_dir: str, stopwords_path: str, tokenizer_name: str = "mecab",
) -> None:

    with open(stopwords_path, "r", encoding="utf-8") as f:
        stopwords = f.read().split("\n")

    tokenizer = get_tokenizer(tokenizer_name)
    nouns_data, token_data, token_pos_data = [], [], []
    for news in tqdm(data):
        keyword, press, category, title, content = news

        # tokenizer를 이용한 tokenizing
        # nouns
        title_nouns = tokenizer.nouns(title)
        content_nouns = tokenizer.nouns(content)
        # tokens & pos_tag
        title_tokens = tokenizer.pos(title)
        content_tokens = tokenizer.pos(content)

        # stopwords 적용
        title_nouns = [word for word in title_nouns if word not in stopwords]
        content_nouns = [word for word in content_nouns if word not in stopwords]
        title_morphs = [word for word, _ in title_tokens if word not in stopwords]
        content_morphs = [word for word, _ in content_tokens if word not in stopwords]
        title_tags = [f"{word}_{pos}" for word, pos in title_tokens if word not in stopwords]
        content_tags = [f"{word}_{pos}" for word, pos in content_tokens if word not in stopwords]

        # append lists
        nouns_data.append((keyword, press, category, title_nouns, content_nouns))
        token_data.append((keyword, press, category, title_morphs, content_morphs))
        token_pos_data.append((keyword, press, category, title_tags, content_tags))

    # save tokens
    with open(f"{save_dir}/nouns_total_data.txt", "wb") as fp:
        pickle.dump(nouns_data, fp)

    with open(f"{save_dir}/tokenized_pos_total_data.txt", "wb") as fp:
        pickle.dump(token_pos_data, fp)

    with open(f"{save_dir}/tokenized_total_data.txt", "wb") as fp:
        pickle.dump(token_data, fp)

    return None
