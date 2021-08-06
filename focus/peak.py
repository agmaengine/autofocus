

def rule_base_search(func):
    step_control_map = {'initial': 5,
                        'fine': 5,
                        'mid': 15,
                        'coarse': 50}
    # f_list = []
    f_previous = 0
    f_max = 0
    x = 0
    x_max = 0
    i = 0
    while True:
        f_current = func(x)
        if i <= 5:
            control = 'initial'
        else:
            if f_current <= 0.25 * f_max:
                control = 'coarse'
                down = 0
            else:
                diff = f_current - f_previous
                if diff > 0.25 * f_previous:
                    control = 'fine'
                    down = 0
                elif control == 'fine' and diff > 0:
                    down = 0
                elif diff < 0:
                    if control == 'fine':
                        down += 1
                    if down == 3:
                        control == 'mid'
                        down = 0
                else:
                    control = 'mid'
                    down = 0
        if f_max < f_current:
            f_max = f_current
            x_max = x
        x = x + step_control_map[control]
        f_previous = f_current
        # f_list.append(f_current)
        i += 1
        if x > 255:
            break
    func(x_max)
    return x_max
