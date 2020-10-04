from Reader import *

import torch
import torch.nn as nn
from collections import Counter
import numpy as np
import json


class RNN(nn.Module):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    min_words_in_sentence = 25
    def InitTrain_x(self):
        print('Инициализирую train_x')

        temp_x = []
        cnt_batch = 0
        cur2 = []
        for row in self.input.data[0]:
            cur = []
            cnt = 0
            for word in row[0]:
                cur.append(self.vocab_to_int[word])
                cnt += 1
                if cnt == RNN.min_words_in_sentence:
                    break

            for _ in range(cnt, RNN.min_words_in_sentence):
                cur.append(0)

            cur2.append(cur)
            cnt_batch += 1
            if cnt_batch % self.batch_size == 0:
                temp_x.append(cur2)
                cur2 = []
        # print(temp_x.shape)
        mem = np.shape(temp_x)
        self.train_x = torch.LongTensor(temp_x)
        print('inputs/train_x.pt инициализирован!')
        torch.save(self.train_x, 'inputs/train_x.pt')
        print('inputs/train_x.pt сохранен в файл!')

    def InitTrain_y(self):
        print('Инициализирую train_y')
        temp_y = []
        cnt_bacth = 0

        cur = []
        for row in self.input.data[0]:
            if row[1] == 'spam':
                cur.append(1)
            else:
                cur.append(0)

            cnt_bacth += 1
            if cnt_bacth % self.batch_size == 0:
                temp_y.append(cur)
                cur = []

        self.train_y = torch.Tensor(temp_y)
        print('inputs/train_y.pt инициализирован!')
        torch.save(self.train_y, 'inputs/train_y.pt')
        print('inputs/train_y.pt сохранен в файл!')

    def InitTest_x(self):
        print('Инициализирую test_x')

        temp_x = []
        cnt_batch = 0
        cur2 = []
        for row in self.input.data[1]:
            cur = []
            cnt = 0
            for word in row[0]:
                cur.append(self.vocab_to_int[word])
                cnt += 1
                if cnt == RNN.min_words_in_sentence:
                    break

            for _ in range(cnt, RNN.min_words_in_sentence):
                cur.append(0)

            cur2.append(cur)
            cnt_batch += 1
            temp_x.append(cur2)
            cur2 = []

        self.test_x = torch.LongTensor(temp_x)
        print('inputs/test_x.pt инициализирован!')
        torch.save(self.test_x, 'inputs/test_x.pt')
        print('inputs/test_x.pt сохранен в файл!')

    def InitTest_y(self):
        print('Инициализирую test_y')
        temp_y = []
        for row in self.input.data[1]:
            if row[1] == 'spam':
                temp_y.append(1)
            else:
                temp_y.append(0)

        self.test_y = torch.Tensor(temp_y)
        print('inputs/test_y.pt инициализирован!')
        torch.save(self.test_y, 'inputs/test_y.pt')
        print('inputs/test_y.pt сохранен в файл!')


    def big_vocab(self):
        print('Начинаю инициализацию словаря')
        vocabs = []
        for row in self.input.data[0]:
            for word in row[0]:
                vocabs.append(word)
        for row in self.input.data[1]:
            for word in row[0]:
                vocabs.append(word)
        vocab_count = Counter(vocabs)
        vocab_count = vocab_count.most_common(len(vocab_count))
        vocab_to_int = {word: index + 2 for index, (word, count) in enumerate(vocab_count)}
        vocab_to_int.update({'__PADDING__': 0})  # index 0 for padding
        vocab_to_int.update({'__UNKNOWN__': 1})  # index 1 for unknown word such as broken character
        vocab_path = 'inputs/big_vocab.json'
        print(f'Сохраняю словарь в файл {vocab_path}')
        json.dump(vocab_to_int, open(vocab_path, 'w'))


    def preprocess(self):
        vocab_path = 'inputs/big_vocab.json'
        try:
            self.vocab_to_int = json.load(open(vocab_path))
            self.vocab_size = len(self.vocab_to_int)
        except Exception as ex:
            print(f'Ошибка при считывании словаря {vocab_path} {type(ex)}')
            if self.input.data == None:
                self.input.Read()
            self.big_vocab()
            self.vocab_to_int = json.load(open(vocab_path))
            self.vocab_size = len(self.vocab_to_int)

    def InitAll(self):
        self.preprocess()
        try:
            self.train_x = torch.load('inputs/train_x.pt')
            print('Файл inputs/train_x.pt загружен!')
        except Exception as ex:
            print('При загрузке файла inputs/train_x.pt произошла ошибка ', type(ex))
            if self.input.data is None:
                self.input.Read()
            self.InitTrain_x()

        try:
            self.train_y = torch.load('inputs/train_y.pt')
            print('Файл inputs/train_y.pt загружен!')
        except Exception as ex:
            print('При загрузке файла inputs/train_y.pt произошла ошибка ', type(ex))
            if self.input.data is None:
                self.input.Read()
            self.InitTrain_y()

        try:
            self.test_x = torch.load('inputs/test_x.pt')
            print('Файл test_x.pt2 загружен!')
        except Exception as ex:
            print('При загрузке файла inputs/test_x.pt произошла ошибка ', type(ex))
            if self.input.data is None:
                self.input.Read()
            self.InitTest_x()

        try:
            self.test_y = torch.load('inputs/test_y.pt')
            print('Файл inputs/test_y.pt загружен!')
        except Exception as ex:
            print('При загрузке файла inputs/test_y.pt произошла ошибка ', type(ex))
            if self.input.data is None:
                self.input.Read()
            self.InitTest_y()

    def __init__(self, input, test = False):
        super().__init__()

        self.input = input
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.batch_size = 80
        if test:
            self.preprocess()
        else:
            self.InitAll()
        self.hidden_size = 25
        self.embedding_dim = 100
        self.num_classes = 1

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, self.num_classes, batch_first=True)
        self.linear_layer = nn.Linear(self.min_words_in_sentence, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        embedded_seq_tensor = self.embedding(x)

        lstm_out, hidden = self.lstm(embedded_seq_tensor, None)
        lstm_out = self.linear_layer(lstm_out[:, :, -1])
        lstm_out = self.sigmoid(lstm_out)

        return lstm_out, hidden

    def test_on_test(self, epoch):
        print('Начинаю тестирование на тестовом датасете!')
        right_answers = 0
        right_spam = 0
        right_fam = 0
        all_spam = 0
        all_fam = 0

        self.test_x = self.test_x.to(RNN.device)
        self.test_y = self.test_y.to(RNN.device)

        cnt = 0
        for inputs in self.test_x:


            outputs, hidden = self(inputs)

            has = 0
            if outputs[0].item() >= 0.5:
                has = 1

            if self.test_y[cnt].item() == 1:
                all_spam += 1
            else:
                all_fam += 1

            if has == self.test_y[cnt].item():
                right_answers += 1
                if has == 1:
                    right_spam += 1
                else:
                    right_fam += 1
            cnt += 1

        with open(f'results/LSTM/epoch{epoch}.txt', 'w') as f:

            f.write(f'На тестовом датасете распознано верно {(right_answers / cnt) * 100}%\n')
            f.write(f'На тестовом датасете информативных текстов верно распознано {(right_spam / all_spam) * 100}%\n')
            f.write(f'На тестовом датасете не информативных текстов верно распознано {(right_fam / all_fam) * 100}%\n')

            print(f'На тестовом датасете распознано верно {(right_answers / cnt) * 100}%')
            print(f'На тестовом датасете информативных текстов верно распознано {(right_spam / all_spam) * 100}%')
            print(f'На тестовом датасете не информативных текстов верно распознано {(right_fam / all_fam) * 100}%')

    @staticmethod
    def train_lstm():
        input = Data()
        neuro = RNN(input).to(RNN.device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(neuro.parameters(), lr=0.03)

        neuro.train_x = neuro.train_x.to(RNN.device)
        neuro.train_y = neuro.train_y.to(RNN.device)
        print('Starting to train')
        neuro.train(True)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
                                                               mode='min', \
                                                               factor=0.5, \
                                                               patience=2)
        for epoch in range(4):
            cnt = 0
            all = len(neuro.train_x)
            print(f'Epoch: {epoch}')

            scheduler.step(epoch)

            for inputs in neuro.train_x:

                outputs, hidden = neuro(inputs)

                loss = criterion(outputs, neuro.train_y[cnt])
                optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(neuro.parameters(), 5)

                optimizer.step()

                cnt += 1
                if cnt % 100 == 0:
                    print(cnt)

            neuro.test_on_test(epoch)

            torch.save(neuro.state_dict(), f'models/lstm_model_epoch{epoch}.pt')


if __name__ == '__main__':
    RNN.train_lstm()
