# Copied from https://github.com/davyzhang/dict-to-protobuf, thanks!
import logging

l = logging.getLogger('dict_to_protbuf')

__all__ = ['dict_to_protobuf']


def parse_list(values, message):
    '''parse list to protobuf message'''
    if isinstance(values[0], dict):  # value needs to be further parsed
        for v in values:
            cmd = message.add()
            parse_dict(v, cmd)
    else:  # value can be set
        message.extend(values)


def parse_dict(values, message):
    for k, v in values.items():
        if isinstance(v, dict):  # value needs to be further parsed
            parse_dict(v, getattr(message, k))
        elif isinstance(v, list):
            parse_list(v, getattr(message, k))
        else:  # value can be set
            try:
                setattr(message, k, v)
            except AttributeError:
                logging.basicConfig()
                l.warning('try to access invalid attributes %r.%r = %r', message, k, v)


def dict_to_protobuf(value, message):
    parse_dict(value, message)
