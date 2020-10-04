from Reader import *
import math
import json
class Bayes:
    def __init__(self, input):
        self.input = input

    def CheckBayes(self, answers, test_ind = 1):
        if test_ind != 0 and test_ind != 1:
            raise AttributeError("0 - выборка для тренировки, 1 - для тестирования")

        samples = 0
        right_spam = 0
        right_fam = 0
        right_answers = 0
        cnt_spam = 0
        cnt_fam = 0
        for index, text in enumerate(self.input.data[test_ind]):
            samples += 1
            if answers[index] == text[1]:
                right_answers += 1
                if answers[index] == 'spam':
                    right_spam += 1
                else:
                    right_fam += 1
            if text[1] == 'spam':
                cnt_spam += 1
            else:
                cnt_fam += 1

        print(f"Общий результат: {(right_answers / samples) * 100 : .2f}%")
        print(f"Верно определено информативных текстов: {(right_spam / cnt_spam) * 100 : .2f}%")
        print(f"Верно определено не информативных текстов: {(right_fam / cnt_fam) * 100 : .2f}%")

        with open('results/Bayes/result.txt', 'w') as f:
            f.write(f"Общий результат: {(right_answers / samples) * 100 : .2f}%\n")
            f.write(f"Верно определено информативных текстов: {(right_spam / cnt_spam) * 100 : .2f}%\n")
            f.write(f"Верно определено не информативных текстов: {(right_fam / cnt_fam) * 100 : .2f}%\n")

        return right_answers / samples

    def bayes_vocab(self):
        print('Начинаю инициализацию bayes_vocab.json')
        vocab = {}
        current = self.input
        for row in current.data[0]:
            for word in row[0]:
                if row[1] == 'spam':
                    if word in vocab:
                        vocab[word][2] += 1
                        vocab[word][0] += 1
                    else:
                        vocab[word] = [1, 0, 1]
                else:
                    if word in vocab:
                        vocab[word][1] += 1
                        vocab[word][0] += 1
                    else:
                        vocab[word] = [1, 1, 0]

        print('Записываю bayes_vocab!')
        json.dump(vocab, open('inputs/bayes_vocab.json', 'w'))

    @staticmethod
    def train_bayes():
        input = Data()
        input.Read()
        bayes = Bayes(input)
        words = None
        try:
            # 0 - сколько слово встречалось везде, 1 - сколько слово встречалось в спаме, 2 - сколько слово встречалось не в спаме
            words = json.load(open('inputs/bayes_vocab.json'))
        except:
            bayes.bayes_vocab()
            words = json.load(open('inputs/bayes_vocab.json'))

        spam_textes = 0
        ham_textes = 0
        all_textes = 0

        # Проходимся по данным для обучения
        for text in input.data[0]:
            all_textes += 1
            if text[1] == 'spam':
                spam_textes += 1
            else:
                ham_textes += 1

        answers = []

        # 0 - тестирование по выборке для обучение, 1 - для тестов
        test_ind = 1
        # Проходимся по тестовым данным
        for text in input.data[test_ind]:
            p_spam = 1
            p_fam = 1
            for word in text[0]:
                if word in words and words[word][1] != 0 and words[word][2] != 0:
                    p_spam += math.log(words[word][2] / words[word][0])
                    p_fam += math.log(words[word][1] / words[word][0])

            p_spam *= spam_textes / all_textes
            p_fam *= ham_textes / all_textes
            cur_ans = 'fam'
            if p_spam > p_fam:
                cur_ans = 'spam'

            answers.append(cur_ans)

        bayes.CheckBayes(answers, test_ind=test_ind)



if __name__ == '__main__':
    Bayes.train_bayes()
