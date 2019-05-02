prefix = 'czech_pdt_'
suffix = '.txt'
inputs = ['train', 'dev']

def intersperse(lst, item):
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result

# TRAIN AND DEV
for input in inputs:
    with open(prefix+input+suffix, 'r', encoding="utf-8") as in_file:
        with open(prefix+'divided_'+input+suffix, 'w', encoding="utf-8") as out_file:
            for line in in_file.readlines():
                parts = line.split('\t')
                if len(parts) == 1:
                    out_file.write(line)
                    continue

                line = parts[:2] + [parts[2][0] + parts[2][1]] + list(parts[2])[2:-1]
                del line[-3:-1]
                parts = intersperse(line, '\t')
                out_file.write(''.join([ch for part in parts for ch in part]) + '\n')


with open(prefix+"test"+suffix, 'r', encoding="utf-8") as in_file:
    with open(prefix+'divided_'+"test"+suffix, 'w', encoding="utf-8") as out_file:
        for line in in_file.readlines():
            parts = line.split('\t')
            if len(parts) == 1:
                out_file.write(line)
                continue
            parts = intersperse(parts[:2] + list('<pad>'*12), '\t')
            out_file.write(''.join([ch for part in parts for ch in part]) + '\n')

