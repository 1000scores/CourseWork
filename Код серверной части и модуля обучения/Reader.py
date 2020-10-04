import os
import random

class Data:
    def __init__(self):
        self.data = None

    def Read(self, percent_test = 0.15):
        #1ый элемент - данные для обучения, 2ой элемент - для тестирования
        length = 0
        answer = []
        cant = 0
        for part in range(1, 7):
            cur_path_ham = f'dataset2/enron{part}/ham/'
            cur_path_spam = f'dataset2/enron{part}/spam/'

            for _, _, files in os.walk(cur_path_ham):
                for file_name in files:
                    file_type = 'fam'
                    with open(cur_path_ham + file_name) as f:
                        try:
                            lst = f.read().splitlines()
                            item = tuple((self.list_to_tokens(lst), file_type))
                            answer.append(item)
                            length += 1
                        except Exception:
                            cant += 1

            for _, _, files in os.walk(cur_path_spam):
                for file_name in files:
                    file_type = 'spam'
                    with open(cur_path_spam + file_name) as f:
                        try:
                            lst = f.read().splitlines()
                            item = tuple((self.list_to_tokens(lst), file_type))
                            answer.append(item)
                            length += 1
                        except Exception:
                            cant += 1


        random.shuffle(answer)

        answer2 = ([], [])
        already_in = 0
        percent_train = 1 - percent_test
        for elem in answer:
            if already_in/length <= percent_train:
                answer2[0].append(elem)
            else:
                answer2[1].append(elem)
            already_in += 1

        len_train = len(answer2[0])
        len_test = len(answer2[1])
        answer = answer2

        print(f'Ошибка при чтении {cant} файлов!')
        print(f"Просмотрено {length} файлов")
        print(f"Добавлено {len(answer[0]) + len(answer[1])} файлов")

        self.data = answer

    # Оставляет только буквы, приводит к нижнему регистру и возвращает список слов
    def list_to_tokens(self, lst):
        answer = []
        for line in lst:
            cur_line = line.lower()

            for elem in cur_line.split(' '):
                if elem == 'subject:' or elem == '':
                    continue
                is_okey = True
                for symbol in elem:
                    if symbol < 'a' or symbol > 'z':
                        is_okey = False
                        break
                if is_okey:
                    answer.append(elem)

        return answer
