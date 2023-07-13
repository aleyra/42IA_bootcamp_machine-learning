class Matrix:
    def __init__(self, obj) -> None:
        # init by values
        if isinstance(obj, list):
            # checks
            nb_line = len(obj)
            if (nb_line == 0):
                raise TypeError("Must be an list of lists of float or int")
            for i in range(nb_line):
                if (not isinstance(obj[i], list)):
                    raise TypeError("Must be an list of lists of float or int")
                nb_col = len(obj[0])
                if (nb_col == 0):
                    raise TypeError("Must be an list of lists of float or int")
                if (len(obj[i]) != nb_col):
                    raise TypeError(
                        "Each list in main list must have the same size"
                    )
                for j in range(len(obj[i])):
                    if (not isinstance(obj[i][j], float)
                        and not isinstance(obj[i][j], int)):
                        raise TypeError(
                            "Must be an list of lists of float or int"
                        )
            # init
            self.shape = (nb_line, nb_col)
            self.data = obj
        # init by shape
        elif isinstance(obj, tuple):
            # checks
            if len(obj) != 2:
                raise TypeError("Shape must be a tuple of 2 int positive")
            if not isinstance(obj[0], int) or not isinstance(obj[1], int):
                raise TypeError("Shape must be a tuple of 2 int > 0")
            if obj[0] < 1 or obj[1] < 1:
                raise TypeError("Shape must be a tuple of 2 int > 0")
            # init
            self.shape = obj
            self.data = []
            for i in range(obj[0]):
                t = []
                for j in range(obj[1]):
                    t.append(0)
                self.data.append(t)
        else:
            raise TypeError("Must init with a list or a tuple")
        
    # add : only matrices of same dimensions.
    def __add__(self, m):
        if not isinstance(m, Matrix):
            raise TypeError("Add is between 2 matrices of same dimensions")
        if self.shape != m.shape:
            raise TypeError("Add is between 2 matrices of same dimensions")
        res = Matrix(m.shape)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                res.data[i][j] = self.data[i][j] + m.data[i][j]
        return res
        
    def __radd__(self, m):
        if not isinstance(m, Matrix):
            raise TypeError("Add is between 2 matrix of same dimensions")
        if self.shape != m.shape:
            raise TypeError("Add is between 2 matrix of same dimensions")
        res = Matrix(m.shape)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                res.data[i][j] = m.data[i][j] + self.data[i][j]
        return res
    
    # sub : only matrices of same dimensions.
    def __sub__(self, m):
        if not isinstance(m, Matrix):
            raise TypeError("Add is between 2 matrix of same dimensions")
        if self.shape != m.shape:
            raise TypeError("Add is between 2 matrix of same dimensions")
        res = Matrix(m.shape)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                res.data[i][j] = self.data[i][j] - m.data[i][j]
        return res
        
    def __rsub__(self, m):
        if not isinstance(m, Matrix):
            raise TypeError("Add is between 2 matrix of same dimensions")
        if self.shape != m.shape:
            raise TypeError("Add is between 2 matrix of same dimensions")
        res = Matrix(m.shape)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                res.data[i][j] = m.data[i][j] - self.data[i][j]
        return res
    
    # div : only scalars.
    def __truediv__(self, l):
        if not isinstance(l, int) and not isinstance(l, float):
            raise TypeError(
                "TrueDiv is only between a matrix and a scalar (int or float)")
        if l == 0:
            raise ZeroDivisionError("Division by zero impossible")
        res = Matrix(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                res.data[i][j] = self.data[i][j] / l
        return res
    
    def __rtruediv__(self, l):
        raise NotImplementedError("Impossible to divise a scalar by a matrix")
    
    # mul : scalars, vectors and matrices , can have errors with vectors and
    # matrices,
    # returns a Vector if we perform Matrix * Vector mutliplication.
    def __mul__(self, l):
        if isinstance(l, int) or isinstance(l, float):  # case scalar
            res = Matrix(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    res.data[i][j] = self.data[i][j] * l
            return res
        # case Vector NOTTODO because done in Vector's class
        elif isinstance(l, Matrix):  # case matrix
            if self.shape[1] != l.shape[0]:
                raise TypeError(
                    "M x N is possible only if M's nb_line = N's nb_col"
                )
            res = Matrix((self.shape[0], l.shape[1]))
            for i in range(res.shape[0]):
                for j in range(res.shape[1]):
                    for k in range(self.shape[1]):
                        res.data[i][j] += self.data[i][k] * l.data[k][j]
            return res
        else:
            raise TypeError(
                """Multiplication is possible between :
    - matrix and scalar
    - matrix and vector
    - matrix and matrix"""
            )
        
    def __rmul__(self, r):
        if isinstance(r, int) or isinstance(r, float):  # case scalar
            res = Matrix(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    res.data[i][j] = r * self.data[i][j]
            return res
        elif isinstance(r, Matrix):  # case matrix
            if r.shape[1] != self.shape[0]:
                raise TypeError(
                    "M x N is possible only if M's nb_line = N's nb_col"
                )
            res = Matrix((r.shape[0], self.shape[1]))
            for i in range(res.shape[0]):
                for j in range(res.shape[1]):
                    for k in range(r.shape[1]):
                        res.data[i][j] += r.data[i][k] * self.data[k][j]
            return res
        else:
            raise TypeError(
                """Multiplication is possible between :
    - scalar and matrix
    - vector and matrix
    - matrix and matrix"""
            )

    def __str__(self) -> str:
        to_print = f"shape = {self.shape}\n["
        for i in range(self.shape[0]):
            to_print += f"\n\t{self.data[i]}"
        to_print += "\n]"
        return to_print
    
    def __repr__(self) -> str:
        to_print = f"["
        for i in range(self.shape[0]):
            to_print += f"\n\t{self.data[i]}"
        to_print += "\n]"
        return to_print
    
    def T(self):
        res = Matrix((self.shape[1],self.shape[0]))
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                res.data[i][j] = self.data[j][i]
        return res

    
class Vector(Matrix):
    def __init__(self, obj) -> None:
        # init by shape
        if isinstance(obj, tuple):
            # checks
            if len(obj) != 2:
                raise TypeError("Shape must be a tuple of 2 int positive")
            if not isinstance(obj[0], int) or not isinstance(obj[1], int):
                raise TypeError("Shape must be a tuple of 2 int positive")
            if (obj[0] != 1 and obj[1] != 1) or obj[0] < 1 or obj[1] < 1:
                raise TypeError("Shape must be (1,n) or (n,1)")
            # init
            self.shape = obj
            self.data = []
            for i in range(self.shape[0]):
                t = []
                for j in range(self.shape[1]):
                    t.append(0)
                self.data.append(t)
        # init by values
        elif isinstance(obj, list):
            if len(obj) != 1:  # case column
                #checks
                for i in range(len(obj)):
                    if not isinstance(obj[i], list):
                        raise TypeError(
                            "the list must contain lists of 1 "
                            + "float or 1 list of several floats"
                        )
                    if len(obj[i]) != 1:
                        raise TypeError(
                            "the list must contain lists of 1 float"
                        )
                    if (
                        not isinstance(obj[i][0], int)
                        and not isinstance(obj[i][0], float)
                    ):
                        raise TypeError(
                            "the list must contain lists of 1 float"
                        )
                #init
                self.data = obj
                self.shape = (len(obj), 1)
            else:  # case line
                # checks
                if not isinstance(obj[0], list):
                    raise TypeError("the list must contain 1 list of floats")
                for i in range(len(obj[0])):
                    if (
                        not isinstance(obj[0][i], int)
                        and not isinstance(obj[0][i], float)
                    ):
                        raise TypeError(
                            "the list must contain 1 list of floats"
                        )
                # init
                self.data = obj
                self.shape = (1, len(obj[0]))
        else:
            raise TypeError("Must init with a list or a tuple")
    
    def __add__(self, v):
        if not isinstance(v, Vector):
            raise TypeError("Add is between 2 vectors of same dimensions")
        if self.shape != v.shape:
            raise TypeError("Add is between 2 vectors of same dimensions")
        res = Vector(v.shape)
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                res.data[i][j] = self.data[i][j] + v.data[i][j]
        return res
    
    def __radd__(self, v):
        if not isinstance(v, Vector):
            raise TypeError("Add is between 2 vectors of same dimensions")
        if self.shape != v.shape:
            raise TypeError("Add is between 2 vectors of same dimensions")
        res = Vector(v.shape)
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                res.data[i][j] = v.data[i][j] + self.data[i][j]
        return res
    
    def __sub__(self, v):
        if not isinstance(v, Vector):
            raise TypeError("Sub is between 2 vectors of same dimensions")
        if self.shape != v.shape:
            raise TypeError("Sub is between 2 vectors of same dimensions")
        res = Vector(v.shape)
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                res.data[i][j] = self.data[i][j] - v.data[i][j]
        return res
    
    def __rsub__(self, v):
        if not isinstance(v, Vector):
            raise TypeError("Sub is between 2 vectors of same dimensions")
        if self.shape != v.shape:
            raise TypeError("Sub is between 2 vectors of same dimensions")
        res = Vector(v.shape)
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                res.data[i][j] = v.data[i][j] - self.data[i][j]
        return res
    
    def __truediv__(self, l):
        if not isinstance(l, int) and not isinstance(l, float):
            raise TypeError(
                "TrueDiv is only between a vector and a scalar (int or float)"
            )
        if l == 0:
            raise ZeroDivisionError("Division by zero impossible")
        res = Vector(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                res.data[i][j] = self.data[i][j] / l
        return res
    
    def __rtruediv__(self, l):
        raise NotImplementedError("Impossible to divise a scalar by a vector")
    
    def __mul__(self, l):
        if isinstance(l, int) or isinstance(l, float):  # case scalar
            res = Vector(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    res.data[i][j] = self.data[i][j] * l
            return res
        elif isinstance(l, Vector):  # case Vector
            if self.shape[0] != 1:
                raise TypeError(
                    "Impossible to do v x u when v is a vector column"
                )
            if l.shape[1] != 1:
                raise TypeError(
                    "Impossible to do v x u when u is a vector line"
                )
            if self.shape[1] != l.shape[0]:
                raise TypeError(
                    "Impossible to do v x u when v's nb_col != u's nb_line"
                )
            res = Vector((1,1))
            for i in range(l.shape[0]):
                res.data[0][0] += self.data[0][i] * l.data[i][0]
            return res
        elif isinstance(l, Matrix):  # case matrix
            if self.shape[0] != 1:
                raise TypeError(
                    "Impossible to do v x M when v is a vector column"
                )
            if self.shape[1] != l.shape[0]:
                raise TypeError(
                    "v x M is possible only if M's nb_line = v's nb_col"
                )
            res = Vector((self.shape[0], l.shape[1]))
            for i in range(res.shape[0]):
                for j in range(res.shape[1]):
                    for k in range(self.shape[1]):
                        res.data[i][j] += self.data[i][k] * l.data[k][j]
            return res
        else:
            raise TypeError(
                """Multiplication is possible between :
    - vector and scalar
    - vector and vector
    - vector and matrix"""
            )

    def __rmul__(self, r):
        if isinstance(r, int) or isinstance(r, float):  # case scalar
            res = Vector(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    res.data[i][j] = r * self.data[i][j]
            return res
        elif isinstance(r, Vector):  # case Vector
            if self.shape[0] != 1:
                raise TypeError(
                    "Impossible to do v x u when v is a vector column"
                )
            if r.shape[1] != 1:
                raise TypeError(
                    "Impossible to do v x u when u is a vector line"
                )
            if r.shape[1] != self.shape[0]:
                raise TypeError(
                    "Impossible to do v x u when v's nb_col != u's nb_line"
                )
            res = Vector((1,1))
            for i in range(self.shape[0]):
                res.data[0][0] += r.data[0][i] * self.data[i][0]
            return res
        elif isinstance(r, Matrix):  # case matrix
            if self.shape[1] != 1:
                raise TypeError(
                    "Impossible to do M x v when v is a vector line"
                )
            if r.shape[1] != self.shape[0]:
                raise TypeError(
                    "v x M is possible only if M's nb_line = v's nb_col"
                )
            res = Vector((r.shape[0], self.shape[1]))
            for i in range(res.shape[0]):
                for j in range(res.shape[1]):
                    for k in range(r.shape[1]):
                        res.data[i][j] += r.data[i][k] * self.data[k][j]
            return res
        else:
            raise TypeError(
                """Multiplication is possible between :
    - scalar and vector
    - vector and vector
    - matrix and vector"""
            )

    def dot(self, v):
        if not isinstance(v, Vector):
            raise TypeError(
                "dot product is possible only between 2 vectors of same shape"
            )
        if v.shape != self.shape:
            raise TypeError(
                "dot product is possible only between 2 vectors of same shape"
            )
        res = 0.
        if v.shape[0] == 1:
            for i in range(v.shape[1]):
                res += self.data[0][i] * v.data[0][i]
        else:
            for i in range(v.shape[0]):
                res += self.data[i][0] * v.data[i][0]
        return res

    def T(self):
        res = Vector((self.shape[1], self.shape[0]))
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                res.data[i][j] = self.data[j][i]
        return res
