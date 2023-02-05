import string

def power_string_generator():
    result = []
    for i in range(11):
        for j in range(11):
            for k in range(11):
                if (i*0.1 + j*0.1 + k*0.1) == 1:
                    result.append([round(i*0.1,2), round(j*0.1,2), round(k*0.1,2)])
    print(result)
    return result

if __name__ == '__main__':
    power_string_generator()