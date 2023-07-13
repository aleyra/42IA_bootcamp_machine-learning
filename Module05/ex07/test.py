from matrix import Matrix
from matrix import Vector

if __name__ == "__main__":
    #Matrices
    #init matrix + __str__ + __repr__
    l1 = [[1.0, 2.0], [3.0, 4.0], [5, 6]]
    m1 = Matrix(l1)
    print(m1)
    m2 = Matrix((3, 3))
    # print(repr(m2))

    #add
    l3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    m3 = Matrix(l3)
    m3x2 = m3 + m3
    # print(m3x2 + m3)
    # print(m1 + m2)# error
    # print(m2 + m1)# error
    # print(m1 + 2)# error
    # print(2 + m1)# error

    #sub
    # print(m3 - m3x2)
    # print(m1 - m3)# error
    # print(m3 - m1)# error
    # print(m1 - 2)# error
    # print(2 - m1)# error

    #truediv
    # print(m1 / 2)
    # print(m1 / m1)# error
    # print(2 / m1)# error

    #mul
    l4 = [[1, 1], [1, 1]]
    m4 = Matrix(l4)
    # print(m1 * 0)
    # print(m1 * m1)# error
    # print(m1 * m3)# error
    # print(m3 * m1)
    # print(m4 * m4)

    #T
    # print(m1)
    m5 = m1.T()
    # print(m5)

    #Vectors
    #init
    lv1 = [[1, 2, 3]]
    v1 = Vector(lv1)
    print(v1)
    lv2 = [[1], [2], [3]]
    v2 = Vector(lv2)
    print(repr(v2))
    tv3 = (1, 2)
    v3 = Vector(tv3)
    # print(v3)

    #add
    v1x2 = v1 + v1
    # print(v1x2)
    # print(type(v1x2))
    # print(v1 + v2)
    # print(v1 + 2)
    # print(v1 + m1)

    #sub
    # print(v1x2 - v1)
    # print(v1 - v1x2)
    # print(type(v1 - v1x2))
    # print(v1 - 1)# error
    # print(1 - v1)# error
    # print(v1 - m1)# error
    # print(m1 - v1)# error

    #truediv
    # print(v1 / 2)
    # print(type(v1 / 2))
    # print(2 / v1)# error
    # print(v1 / v1)# error
    # print(v1 / m1)# error
    # print(m1 / v1)# error

    #mul
    # print(v1 * 2)
    # print(type(v1 * 2))
    # print(2 * v1)
    # print(v1 * v1)# error
    # print(v1 * v2)
    # print(v1 * m1)
    # print(m1 * v1)# error
    # print(v2 * m1)# error
    # print(m1 * v2)# error
    # print(v2 * m1)# error
    # print(m5 * v2)
    # print(v2 * v1)# error
    # print(v2 * v2)# error

    #dot
    # print(v1.dot(v1))
    # print(v1.dot(v2))# error
    # print(v1.dot(2))# error

    #T
    # print(v1.T())
    # print(type(v1.T()))