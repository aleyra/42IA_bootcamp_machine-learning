class Matrix:
    def __init__(self, obj) -> None:
        # checks
        print(obj)
        if (isinstance(obj, list)):     
            print("check 1 ok")
            nb_line = len(obj)
            if (nb_line == 0):
                raise TypeError("Must be an list of lists of float or int")
            for i in range(nb_line):
                if (not isinstance(obj[i], list)):
                    raise TypeError("Must be an list of lists of float or int")
                print("check 2 ok")
                nb_col = len(obj[0])
                if (nb_col == 0):
                    raise TypeError("Must be an list of lists of float or int")
                print("check 3 ok")
                if (len(obj[i]) != nb_col):
                    raise TypeError(
                        "Each list in main list must have the same size")
                print("check 4 ok")
                for j in range(len(obj[i])):
                    if (not isinstance(obj[i][j], float)
                        and not isinstance(obj[i][j], int)):
                        raise TypeError(
                            "Must be an list of lists of float or int")
                    print("check 5 ok")
            self.shape = (nb_line, nb_col)
        if (isinstance(obj, tuple)):
            print("check 6 ok")
            if (len(obj) != 2):
                raise TypeError("Shape must be a tuple of 2 int positive")
            print("check 7 ok")
            if (not isinstance(obj[0], int) or not isinstance(obj[1], int)):
                raise TypeError("Shape must be a tuple of 2 int positive")
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
        if (not isinstance(m, Matrix)):
            raise TypeError("Add is between 2 matrix of same dimensions")
        if (self.shape != m.shape):
            raise TypeError("Add is between 2 matrix of same dimensions")
        res = Matrix(m.shape)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                res.data[i][j] = self.data[i][j] + m.data[i][j]
        return res
        
    def __radd__(self, m):
        if (not isinstance(m, Matrix)):
            raise TypeError("Add is between 2 matrix of same dimensions")
        if (self.shape != m.shape):
            raise TypeError("Add is between 2 matrix of same dimensions")
        res = Matrix(m.shape)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                res.data[i][j] = m.data[i][j] + self.data[i][j]
        return res
    
    # sub : only matrices of same dimensions.
    def __sub__(self, m):
        if (not isinstance(m, Matrix)):
            raise TypeError("Add is between 2 matrix of same dimensions")
        if (self.shape != m.shape):
            raise TypeError("Add is between 2 matrix of same dimensions")
        res = Matrix(m.shape)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                res.data[i][j] = self.data[i][j] - m.data[i][j]
        return res
        
    def __rsub__(self, m):
        if (not isinstance(m, Matrix)):
            raise TypeError("Add is between 2 matrix of same dimensions")
        if (self.shape != m.shape):
            raise TypeError("Add is between 2 matrix of same dimensions")
        res = Matrix(m.shape)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                res.data[i][j] = m.data[i][j] - self.data[i][j]
        return res
    
    # div : only scalars.
    def __truediv__(self, l):
        if (not isinstance(l, int) and not isinstance(l, float)):
            raise TypeError(
                "TrueDiv is only between a matrix and a scalar (int or float)")
        if (l == 0):
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
        if (isinstance(l, int) or isinstance(l, float)):
            res = Matrix(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    res.data[i][j] = self.data[i][j] * l
            return res
        #case Vector TODO
        elif (isinstance(l, Matrix)):
            if (self.shape[0] != l.shape[1]):
                raise TypeError(
                    "M x N is possible only if M's nb_line = N's nb_col")
            res = Matrix((self.shape[0], self.shape[0]))
            for i in range(self.shape[0]):
                for j in range(self.shape[0]):
                    for k in range(self.shape[1]):
                        res.data[i][j] += self.data[i][k] * l.data[k][j]
            return res
        else:
            raise TypeError("""Multiplication is possible between :
        - matrix and scalar
        - matrix and vector
        - matrix and matrix""")
        
    def __rmul__(self, r):
        if (isinstance(r, int) or isinstance(r, float)):
            res = Matrix(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    res.data[i][j] = r * self.data[i][j]
            return res
        #case Vector TODO
        elif (isinstance(r, Matrix)):
            if (r.shape[0] != self.shape[1]):
                raise TypeError(
                    "M x N is possible only if M's nb_line = N's nb_col")
            res = Matrix((r.shape[0], r.shape[0]))
            for i in range(r.shape[0]):
                for j in range(r.shape[0]):
                    for k in range(r.shape[1]):
                        res.data[i][j] += r.data[i][k] * self.data[k][j]
            return res
        else:
            raise TypeError("""Multiplication is possible between :
        - scalar and matrix
        - vector and matrix
        - matrix and matrix""")

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