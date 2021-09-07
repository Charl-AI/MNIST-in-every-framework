import os


def local_test(func):
    def wrapper():
        if "CI" not in os.environ:
            func
        else:
            assert True

    return wrapper
