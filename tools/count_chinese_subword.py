import sys

def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
        return True
    else:
        return False

def is_contain_chinese(subword):
    for each in subword:
        if is_chinese(each):
            return True
    return False


if __name__ == '__main__':
    with open(sys.argv[1], 'r', encoding='utf-8') as fin:
        data = fin.readlines()

    sum_subword = 0
    sum_chinese_subword = 0
    for line in data:
        for subword in line.strip().split(' '):
            sum_subword += 1
            if is_contain_chinese(subword):
                sum_chinese_subword += 1

    
    print('CHINESE / SUM:', sum_chinese_subword, sum_subword)