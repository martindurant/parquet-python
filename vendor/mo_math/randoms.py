# encoding: utf-8
#

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import random
import string

SIMPLE_ALPHABET = string.ascii_letters + string.digits
SEED = random.Random()


class Random(object):
    @staticmethod
    def set_seed(seed):
        globals()["SEED"] = random.Random(seed)

    @staticmethod
    def string(length, alphabet=SIMPLE_ALPHABET):
        result = ''
        for i in range(length):
            result += SEED.choice(alphabet)
        return result

    @staticmethod
    def hex(length):
        return Random.string(length, string.digits + 'ABCDEF')

    @staticmethod
    def base64(length):
        return Random.string(length, string.digits + string.letters + '+/')

    @staticmethod
    def int(*args):
        return SEED.randrange(*args)

    @staticmethod
    def range(start, stop, *args):
        return SEED.randrange(start, stop, *args)

    @staticmethod
    def float(*args):
        if args:
            return SEED.random()*args[0]
        else:
            return SEED.random()

    @staticmethod
    def sample(data, count):
        num = len(data)
        return [data[Random.int(num)] for i in range(count)]

    @staticmethod
    def combination(data):
        output = []
        data = list(data)
        num = len(data)
        for i in range(num):
            n = Random.int(num-i)
            output.append(data[n])
            del data[n]
        return output

    @staticmethod
    def bytes(count):
        output = bytearray(SEED.randrange(256) for i in range(count))
        return output

    @staticmethod
    def weight(weights):
        """
        RETURN RANDOM INDEX INTO WEIGHT ARRAY, GIVEN WEIGHTS
        """
        total = sum(weights)

        p = SEED.random()
        acc = 0
        for i, w in enumerate(weights):
            acc += w / total
            if p < acc:
                return i
        return len(weights) - 1
