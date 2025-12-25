import fasttext

model_path_nsfw = "jigsaw_fasttext_bigrams_nsfw_final.bin"
model_nsfw = fasttext.load_model(model_path_nsfw)

def classify_nsfw(string: str) -> tuple[str, float]:
    string = string.replace('\n', ' ')
    labels, scores = model_nsfw.predict(string, k=1)
    label = labels[0].replace('__label__', '')
    score = scores[0]
    return (label, score)


# model_path_hatespeech = "jigsaw_fasttext_bigrams_hatespeech_final.bin"
# model_hatespeech = fasttext.load_model(model_path_hatespeech)