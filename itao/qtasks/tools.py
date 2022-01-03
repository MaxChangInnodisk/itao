from typing import Optional, Union


from typing import Tuple

def parse_arguments(key_args:list, in_args:dict) -> Tuple[bool, dict, dict]:
    
    ret = False
    new_args, error_args = dict(), dict()
    
    for key in key_args:
        for in_key, in_val in in_args.items():
            if in_key==key:
                new_args[key]=in_val
            else:
                error_args[key]=None
                ret=True

    return ret, new_args, error_args
        