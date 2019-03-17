import glob
import json


def get_all_options(a, output):
    if a == 0:
        return
    if len(output) == 0:
        output.append([0])
        output.append([1])
        output.append([2])
        output.append([3])
        return get_all_options(a - 1, output)
    item_len = len(output)
    for idx in range(item_len):
        temp = output.pop(0)
        for i in range(4):
            temp.append(i)
            output.append(temp[:])
            temp.pop(-1)
    return get_all_options(a - 1, output)


file_name_list = glob.glob("./CLOTH/train/*/*.json")
mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
out_file = open("./train.txt", "w")

out = []
progress = 0
for file_name in file_name_list:
    progress += 1
    if progress % 100 == 0:
        print(progress)
    data = json.load(open(file_name, "rb"))
    article = data['article'].replace("  ", " ").replace(",", " ").replace('"', " ").replace("?", ".").replace("\t",
                                                                                                               " ")
    options = data['options']
    answers = [mapping[i] for i in data['answers']]
    ops_start = 0
    id = file_name.split("/")[-1].split(".")[0]
    for sentence in filter(lambda x: len(x) > 0, [i.strip() for i in article.split(".")]):
        num_option = sentence.count("_")
        if num_option > 0:
            ops_end = ops_start + num_option
            option_ans = answers[ops_start: ops_end]
            all_ops = []
            get_all_options(num_option, all_ops)

            for ops in all_ops:
                sample = list(filter(lambda x: len(x) > 0, sentence.split(" ")))
                counter = 0
                for i, token in enumerate(sample):
                    if token == "_":
                        sample[i] = options[ops_start + counter][ops[counter]]
                        counter += 1
                proceed_sentenc = " ".join(sample)
                answer = 0
                if ops == option_ans:
                    answer = 1

                out_file.write("{}|||{}|||{}".format(id, proceed_sentenc, answer))
                out_file.write("\n")
            ops_start = ops_end
