import re

def remove_hashtags(text):
    clean_text = re.sub(r'#[A-Za-z0-9_]+', "", text)
    return clean_text


def remove_user_mentions(text):
    clean_text = re.sub(r'@[A-Za-z0-9_]+', "", text)
    return clean_text


def remove_urls(text):
    clean_text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    return clean_text


preprocessing_setps = {
    'remove_hashtags': remove_hashtags,
    'remove_urls': remove_urls,
    'remove_user_mentions': remove_user_mentions
}

def process_text(steps, text):
    if steps is not None:
        for step in steps:
            text = preprocessing_setps[step](text)

    processed_text = ""
    for tx in text:
        processed_text += tx + " "
    return processed_text