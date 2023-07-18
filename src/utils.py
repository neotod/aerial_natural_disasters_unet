def convert_number_to_4_digits_str(num):
    leading_zeros = ['0' for _ in range(4 - len(str(num)))]
    leading_zeros += list(str(num))

    return ''.join(leading_zeros)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']