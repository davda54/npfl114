# předzpracování datasetu -- rozdělení tagu na jednotlivé podtagy
# první dva podtagy sloučeny do jednoho, protože druhý je specifikací prvního
# 13. a 14. tag není používán
# 12. tag je v datasetu plně určen z prvních dvou

# první dva tagy často vylučují použití dalších tagů, ty jsou proto maskovány pomocí znaku <pad>



prefix = 'czech_pdt_'
suffix = '.txt'
inputs = ['train', 'dev']

def intersperse(lst, item):
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result


forms = {}
for input in inputs:
    with open(prefix+input+suffix, 'r', encoding="utf-8") as in_file:
        for line in in_file.readlines():
            parts = line.split('\t')
            if len(parts) == 1: continue
            tag = parts[2]
            POS = tag[:2]
            if POS not in forms:
                forms[POS] = list(tag)
                forms[POS] = forms[POS][:11] + [forms[POS][14]]
                continue

            form = forms[POS]
            tag = tag[:11] + tag[14]
            for i, c in enumerate(tag):
                if form[i] == c: continue
                if c == '-' or form[i] == '-': form[i] = '*'; continue
                form[i] = 'X'
            forms[POS] = form


# with open("tags2.txt", "w", encoding="utf-8") as out_file:
#     for form in sorted(forms.values()):
#         print(''.join(form[:12] + [form[14]]), file=out_file)
#     print(file=out_file)
#     print(''.join([str(int(c)) for i,c in enumerate(possible_non) if i not in[12,13]]), file=out_file)
# exit()

# TRAIN AND DEV
for input in inputs:
    with open(prefix+input+suffix, 'r', encoding="utf-8") as in_file:
        with open(prefix+'sem_divided_'+input+suffix, 'w', encoding="utf-8") as out_file:
            for line in in_file.readlines():
                parts = line.split('\t')
                if len(parts) == 1:
                    out_file.write(line)
                    continue

                POS = parts[2][:2]
                line = parts[:2] + [POS] + list(parts[2])[2:-1]
                del line[-4:-1]
                form = forms[POS]
                for i,c in enumerate(form):
                    if i < 2: continue
                    if c == '-': line[1 + i] = '<pad>'
                parts = intersperse(line, '\t')
                out_file.write(''.join([ch for part in parts for ch in part]) + '\n')


with open(prefix+"test"+suffix, 'r', encoding="utf-8") as in_file:
    with open(prefix+'sem_divided_'+"test"+suffix, 'w', encoding="utf-8") as out_file:
        for line in in_file.readlines():
            parts = line.split('\t')
            if len(parts) == 1:
                out_file.write(line)
                continue
            parts = intersperse(parts[:2] + list('<pad>'*12), '\t')
            out_file.write(''.join([ch for part in parts for ch in part]) + '\n')

