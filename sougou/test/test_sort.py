arr = [4, 3, 2, 1]


def Bubble(array):
    step = 0

    array_len = len(array)
    for i in range(0, array_len - 1, 1):
        for j in range(array_len - 1, i, -1):
            step += 1
            if array[j] < array[j - 1]:
                temp = array[j]
                array[j] = array[j - 1]
                array[j - 1] = temp

            print('step: %d %s' % (step, arr))
        print('----------------------------')
    return array


# print(Bubble(arr))


def quick_sort(arr):
    """
    快速排序
    :param arr:
    :return:
    """
    half = len(arr) / 2
    