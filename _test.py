_list = [10, 1, 9, 2, 8, 3, 7, 4, 6, 5, 5]
k = 3


def quick_sort(_arr):
    global k
    if len(_arr) <= 1:
        return _arr
    piovt = _arr[len(_arr) // 2]
    left = [x for x in _arr if x < piovt]
    middle = [x for x in _arr if x == piovt]
    right = [x for x in _arr if x > piovt]
    if len(right) > k:
        return quick_sort(right)
    elif len(right) == k:
        return min(right)
    else:
        k -= len(right)
        if len(middle) > k:
            return middle[0]
        else:
            return quick_sort(left)


print(quick_sort(_list))
