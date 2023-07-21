# Module 05
## exercice 00
Notons que ce sont les stats anglaises -_-

## exercice 01
liens utiles
* ajouter une col à une pos specifique : https://www.statology.org/numpy-add-column/
* type de np.arr : https://stackoverflow.com/questions/36783921/valueerror-when-checking-if-variable-is-none-or-numpy-array
* check np.arr empty : https://stackoverflow.com/questions/11295609/how-can-i-check-whether-a-numpy-array-is-empty-or-not

## exercice 02
liens utiles
* multi mat : https://numpy.org/doc/stable/reference/generated/numpy.matmul.html

## exercice 05
liens utiles
* abs() : https://www.w3schools.com/python/python_math.asp
* make sklearn.metrics work : https://scikit-learn.org/stable/install.html
-> `conda install scikit-learn`

## exercice 07
liens utiles
* multi constructor : https://realpython.com/python-multiple-constructors/#instantiating-classes-in-python
* heritage : https://www.pierre-giraud.com/python-apprendre-programmer-cours/oriente-objet-heritage-polymorphisme/
* repr : https://www.w3resource.com/python/built-in-function/repr.php

# Module 06
## exercice 00
liens utiles
* np.transpose : https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
* multiply 2 matrices : https://numpy.org/doc/stable/reference/generated/numpy.matmul.html

## exercice 05
liens utiles
* np.min : https://numpy.org/doc/stable/reference/generated/numpy.ndarray.min.html

# Module 07
## exercice 02
pb dans les theta donnés, ils ne sont pas de la bonne dimension.
donc tests impossible a effectuer...
enfin faudrait retrouver la 4e coord de theta. Faisable mais long.
pour theta2 c'est (0, 0, 0, 0)
pour theta1 ce serait (3, 0.5, -6, -3.059813084)... ouais y'a moyen que ce soit un truc du genre
c'est tout theta1 qui va pas...


## exercice 04
attention, le résultat donné est en écriture scientifique... et pas l'output attendu...

## exercice 05
### Part 2
il faut faire le mse entre y et y_hat, et non pas entre x et y... donc on doit d'abord obtenir y_hat avec predict
pour i = 19025,
theta = [
        [3.90663578e+299]
        [4.49758795e+300]
        [4.53451611e+301]
        [3.62457311e+301]]
!!!
tu m'étonnes que j'ai un overflow

## exercice 06
liens utiles
* for i in [a, b] : https://www.w3schools.com/python/python_for_loops.asp
* add column : https://www.geeksforgeeks.org/python-ways-to-add-row-columns-in-numpy-array/
* vector to power : https://numpy.org/doc/stable/reference/generated/numpy.power.html
