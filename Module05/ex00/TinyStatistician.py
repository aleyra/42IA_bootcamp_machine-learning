import math

class TinyStatistician:
	# def __init__(lst, data) -> None:
	# 	if (not isinstance(data, list) and not isinstance(data, tuple)):
	# 		raise AttributeError("data must be a list or a tuple")
	# 	lst = data

	def mean(self, lst):
		if (len(lst) == 0):
			return None
		m = 0
		for i in range(len(lst)):
			m += lst[i]
		m = m / len(lst)
		return m

	def median(self, lst):
		l = len(lst)
		if l == 0:
			return None
		s = lst.sort()
		if l % 2 == 0:
			return lst[int(l / 2 - 1)]
		else:
			return lst[int((l + 1) / 2 - 1)]
		
	def quartile(self, lst):
		if (len(lst) == 0):
			return None
		q1f = len(lst) / 4
		q3f = 3 * q1f
		ret = [float(lst[math.floor(q1f)]), float(lst[math.floor(q3f)])]
		return ret
	
	def var(self, lst):
		if (len(lst) == 0):
			return None
		s2 = 0
		m = self.mean(lst)
		for i in range(len(lst)):
			s2 += math.pow(lst[i] - m, 2)
		s2 /= len(lst) - 1
		return s2
	
	def std(self, lst):
		if (len(lst) == 0):
			return None
		s2 = self.var(lst)
		return math.sqrt(s2)

if __name__ == "__main__":
	tstat = TinyStatistician()
	a = [1, 42, 300, 10, 59]
	print(tstat.mean(a))
	# Expected result: 82.4
	print(tstat.median(a))
	# Expected result: 42.0
	print(tstat.quartile(a))
	# Expected result: [10.0, 59.0]
	print(tstat.var(a))
	# Expected result: 12279.439999999999
	print(tstat.std(a))
	# Expected result: 110.81263465868862