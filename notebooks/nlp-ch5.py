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
                wf.write(f'{w.text}\t{w.lemma}\t{w.upos}\t{w.xpos}\t{w.feats}'
                         f'\t{w.governor}\t{w.dependency_relation}\n')
        wf.write('---\n')
        if sent_count % 1000 == 0:
            print(f'Processed {sent_count} sentences...')
            break
print(f'Process done. {sent_count} sentences, {time.time() - start} sec.')

#%% [markdown]
# 係り受け解析結果ファイルの内容：
# ```
# 吾輩	吾輩	NOUN	_	_	3	nsubj
# は	は	ADP	_	_	1	case
# 猫	猫	NOUN	_	_	0	root
# である	だ	AUX	_	_	3	cop
# 。	。	PUNCT	_	_	3	punct
# ---
# 名前	名前	NOUN	_	_	4	nsubj
# は	は	ADP	_	_	1	case
# まだ	まだ	ADV	_	_	4	advmod
# 無い	無い	ADJ	_	_	0	root
# 。	。	PUNCT	_	_	4	punct
# ---
# ...
# ```

#%% [markdown]
# ## 40. 係り受け解析結果の読み込み（形態素）
#
# 形態素を表すクラスMorphを実装せよ．
# このクラスは表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）を
# メンバ変数に持つこととする．
# さらに，CaboChaの解析結果（neko.txt.cabocha）を読み込み，
# 各文をMorphオブジェクトのリストとして表現し，3文目の形態素列を表示せよ．

#%%
from dataclasses import dataclass


@dataclass
class Morph:
    '''形態素を表すクラス'''
    surface: str
    base: str
    pos: str
    pos1: str


with open('data/neko.txt.snd', 'r', encoding='utf-8') as f:
    count = 1
    sent = []
    for line in f:
        line = line.strip()
        if line == '---':
            if count == 3:
                print(sent)
                break
            else:
                count = count + 1
                sent.clear()
                continue
        elms = line.strip().split('\t')
        sent.append(
            Morph(surface=elms[0], base=elms[1], pos=elms[2], pos1=None))
    print(sent)

#%% [markdown]
# ## 41. 係り受け解析結果の読み込み（文節・係り受け）
# 40に加えて，文節を表すクラスChunkを実装せよ．
# このクラスは形態素（Morphオブジェクト）のリスト（morphs），
# 係り先文節インデックス番号（dst），
# 係り元文節インデックス番号のリスト（srcs）をメンバ変数に持つこととする．
# さらに，入力テキストのCaboChaの解析結果を読み込み，
# １文をChunkオブジェクトのリストとして表現し，8文目の文節の文字列と係り先を表示せよ．
# 第5章の残りの問題では，ここで作ったプログラムを活用せよ．

