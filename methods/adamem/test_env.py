import ray
ray.init()
@ray.remote(resources={'webshop_env': 1})
def test():
    return 'env ok'
print(ray.get(test.remote()))
