#%% [markdown]
# # 第5章: 係り受け解析
# 夏目漱石の小説『吾輩は猫である』の文章（neko.txt）をCaboChaを使って係り受け解析し，その結果をneko.txt.cabochaというファイルに保存せよ．このファイルを用いて，以下の問に対応するプログラムを実装せよ．
#%% [markdown]
# `neko.txt`はdataディレクトリに格納されている。
# このファイルをstanfordnlpで構文解析し、その結果を`data/neko.txt.snd`に出力する
#%%
import time

from stanfordnlp import Pipeline

start = time.time()
nlp = Pipeline(lang='ja')
print(f'Init done. {time.time() - start} sec.')
#%%
start = time.time()
line_count = 0
sent_count = 0
with open(
        'data/neko.txt', encoding='utf-8') as f, open(
            'data/neko.txt.snd', 'w', encoding='utf-8') as wf:
    for line in f:
        line_count = line_count + 1
        if line_count < 3:
            #本文は3行目から開始
            continue
        sentence = line.strip()
        if len(sentence) == 0:
            continue
        sent_count = sent_count + 1
        doc = nlp(sentence)
        for s in doc.sentences:
            for w in s.words:
                wf.write(
                    f'{w.text}\t{w.lemma}\t{w.upos}\t{w.xpos}\t{w.feats}\t{w.governor}\t{w.dependency_relation}\n'
                )
        wf.write('---\n')
        if sent_count % 1000 == 0:
            print(f'Processed {sent_count} sentences...')
            break
print(f'Process done. {sent_count} sentences, {time.time() - start} sec.')

#%% [markdown]
# ## 40. 係り受け解析結果の読み込み（形態素）
#
# 形態素を表すクラスMorphを実装せよ．このクラスは表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をメンバ変数に持つこととする．さらに，CaboChaの解析結果（neko.txt.cabocha）を読み込み，各文をMorphオブジェクトのリストとして表現し，3文目の形態素列を表示せよ．

#%%
from dataclasses import dataclass
from stanfordnlp.pipeline.doc import Word


@dataclass
class Morph:
    '''形態素を表すクラス'''
    surface: str
    base: str
    pos: str
    pos1: str

    def __init__(self, word: Word):
        self.surface = word.text
        self.base = word.lemma
        self.pos = word.upos
        self.pos1 = word.upos  # 品詞細分類に相当する情報がないので、とりあえずUniversal POSを入れておく


doc = nlp("太郎は犬に団子をあげ、猫にはあげなかった。")
morphs = [Morph(w) for w in doc.sentences[0].words]
print(morphs)
