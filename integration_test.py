from pytile.tile import *
from tests.test_basic_ops import *
from tests.test_matrix_ops import *
from tests.test_vector_ops import *

def run_all_tests():
    test_load_store()
    test_scalar_ops()
    test_matmul()
    test_dlt() 
    test_vec_add()
    test_composite_ops()
    print("所有测试通过！")

if __name__ == "__main__":
    run_all_tests()