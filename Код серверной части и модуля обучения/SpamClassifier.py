#!/usr/bin/env python3
import sys
import codecs
import cgi



import argparse
import json
from LSTM import RNN
import torch
import os
import math

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
print('Content-type: text/plain; charset=utf-8\n\n')

def list_to_tokens(lst):
    answer = []
    for line in lst:
        cur_line = line.lower()

        for elem in cur_line.split(' '):
            if elem == 'subject:' or elem == '':
                continue
            is_okey = True
            for ind, symbol in enumerate(elem):
                if symbol < 'a' or symbol > 'z':
                    if ind == len(elem)-1:
                        continue
                    is_okey = False
                    break
            if is_okey:
                answer.append(elem)

    return answer


def LSTM(cur_input):
    inp = None
    lst = cur_input.splitlines()
    inp = list_to_tokens(lst)

    vocab_to_int = None
    try:
        vocab_to_int = json.load(open('inputs/big_vocab.json'))
    except Exception:
        wasExcept = 'Ошибка при чтении словаря inputs/big_vocab.json'
        print(wasExcept)
        return wasExcept
    

    def InitTest_x(inp):

        temp_x = []
        cur = []
        cnt = 0
        for word in inp:
            if word in vocab_to_int:
                cur.append(vocab_to_int[word])
                cnt += 1
            if cnt == RNN.min_words_in_sentence:  # min_words_in_sentence
                break

        for _ in range(cnt, RNN.min_words_in_sentence):
            cur.append(0)

        temp_x.append(cur)
        # print(temp_x.shape)
        test_x = torch.LongTensor(temp_x)

        return test_x

    test_x = InitTest_x(inp)
    BIG_MACHINE = None
    try:
        BIG_MACHINE = RNN(inp, test = True)
        BIG_MACHINE.load_state_dict(torch.load('model/lstm_model.pt'))
    except Exception as ex:
        wasExcept = 'Ошибка при чтении модели model/lstm_model.pt'
        print(wasExcept)
        return wasExcept


    machine_ans = BIG_MACHINE(test_x)

    if machine_ans[0] >= 0.5:
        machine_ans = 'Похоже на не информативный текст'
    else:
        machine_ans = 'Похоже на информативный текст!'

    return machine_ans

def bayes_predict_single(cur_input):
    lst = cur_input.splitlines()
    text = list_to_tokens(lst)

    words = None
    try:
        words = json.load(open('inputs/bayes_vocab.json'))
    except Exception as ex:
        return 'Ошибка при чтении словаря для Байеса'

    p_spam = 0
    p_ham = 0
    for word in text:
        if word in words and words[word][1] != 0 and words[word][2] != 0:
            p_spam += math.log(words[word][2] / words[word][0])
            p_ham += math.log(words[word][1] / words[word][0])
    ans = 'Похоже на информативный текст!'
    if p_spam > p_ham:
        ans = 'Похоже на не информативный текст'
    return ans

def run():
    cur_input = None
    form = cgi.FieldStorage()
    if "input" in form:
        cur_input = form["input"].value
    else:
        print("Did not get input, try again!")
        return

    print(f"LSTM: {cur_input[:10]}...: {LSTM(cur_input)}")

    cur_ans = bayes_predict_single(cur_input)
    print(f"Bayes: {cur_input[:10]}...: {cur_ans}")



run()
