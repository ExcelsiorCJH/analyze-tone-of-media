import re
import glob
import pickle

from collections import defaultdict, Counter
from tqdm import tqdm
from utils.types_ import *


def data_merge(dir_path: str) -> Dict:
    data_paths = glob.glob(f"{dir_path}/*/*.txt")
    data_paths = sorted(data_paths)
    data_dict = defaultdict(list)

    for data_path in tqdm(data_paths):
        keyword = data_path.split("/")[-2]
        with open(data_path, "rb") as fp:
            data_dict[keyword].extend(pickle.load(fp))

    return data_dict


def data_clean(data_dict: Dict) -> List[Tuple]:
    clean_data = []
    for key, articles in data_dict.items():
        for article in tqdm(articles, desc=f"{key} data_clean"):
            press, cat, title, content = article
            if press in ["조선일보", "동아일보", "경향신문", "한겨레"]:
                title = clean_text(title)
                content = clean_text(content)

                clean_data.append((key, press, cat, title, content))

    return clean_data


def build_vocab(dataset: List[Tuple], save_dir: str, num_words: int = 30000) -> Union[Dict, Dict]:
    # 1. tokenization
    all_tokens = []
    for data in tqdm(dataset):
        all_tokens.extend(data[4])

    # 2. build vocab
    vocab = Counter(all_tokens)
    vocab = vocab.most_common(num_words)

    # 3. add pad & unk tokens
    word_index = defaultdict()
    word_index["<PAD>"] = 0
    word_index["<UNK>"] = 1

    for idx, (word, _) in enumerate(vocab, 2):
        word_index[word] = idx

    index_word = {idx: word for word, idx in word_index.items()}

    with open(f"{save_dir}/word_index.pkl", "wb") as f:
        dill.dump(word_index, f)

    return word_index, index_word


def clean_text(text: str) -> str:
    """기사 내용 전처리 함수
    Args:
        - text: str 형태의 텍스트
    Return:
        - text: 전처리된 텍스트"""
    # Common
    text = re.sub("\n", " ", text)
    # E-mail 제거#
    text = re.sub("([\w\d.]+@[\w\d.]+)", "", text)
    text = re.sub("([\w\d.]+@)", "", text)
    # 괄호 안 제거#
    text = re.sub("<[\w\s\d‘’=/·~:&,`]+>", "", text)
    text = re.sub("\([\w\s\d‘’=/·~:&,`]+\)", "", text)
    text = re.sub("\[[\w\s\d‘’=/·~:&,`]+\]", "", text)
    text = re.sub("【[\w\s\d‘’=/·~:&,`]+】", "", text)
    text = re.sub("\(.*\)", "", text)
    text = re.sub("\[.*\]", "", text)

    # 전화번호 제거#
    text = re.sub("(\d{2,3})-(\d{3,4}-\d{4})", "", text)  # 전화번호
    text = re.sub("(\d{3,4}-\d{4})", "", text)  # 전화번호
    # 홈페이지 주소 제거#
    text = re.sub("(https:)", "", text)
    text = re.sub("(www.\w.+)", "", text)
    text = re.sub("(.\w+.com)", "", text)
    text = re.sub("(.\w+.co.kr)", "", text)
    text = re.sub("(.\w+.go.kr)", "", text)
    # 기자 이름 제거#
    text = re.sub("/\w+[=·\w@]+\w+\s[=·\w@]+", "", text)
    text = re.sub("\w{2,4}\s기자", "", text)
    # 한자 제거#
    text = re.sub("[\u2E80-\u2EFF\u3400-\u4DBF\u4E00-\u9FBF\uF900]+", "", text)
    # 특수기호 제거#
    text = re.sub("[◇#/▶▲◆■●△①②③★○◎▽=▷☞◀ⓒ□?㈜♠☎]", "", text)
    # 따옴표 제거#
    text = re.sub("[\"'”“‘’]", "", text)
    # 2안_숫자제거#
    # text = regex.sub('[0-9]+',"",text)

    text = " ".join(text.split())
    return text

