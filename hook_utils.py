

def get_module(input_string):
    split_string = input_string.split('.')
    converted_string = ''
    for element in split_string:
        if element.isdigit():
            converted_string += f"[{element}]"
        else:
            converted_string += f".{element}"
    converted_string = converted_string[1:]
    return converted_string

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def set_hooks(model, positions, pipe):
    for block in positions:
        name = block
        ext = get_module(block) + f".register_forward_hook(get_activation('" + name + "'))"
        print("pipe.unet." + ext)
        exec("pipe.unet." + ext)
