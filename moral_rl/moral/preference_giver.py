import numpy as np
import math
import scipy.stats as st


def check_pareto_dom(ret_a, ret_b):
	pareto_dom_a = ret_a >= ret_b
	pareto_dom_b = ret_b >= ret_a
	return pareto_dom_a.all() and not pareto_dom_b.all()


class TargetGiverv3:
	def __init__(self, target):
		self.target = np.array(target)

	def query_pair(self, ret_a, ret_b):
		dist_a = ((ret_a-self.target)**2).sum()
		dist_b = ((ret_b-self.target)**2).sum()

		if dist_a < dist_b:
			return [1, 0]
		elif dist_b < dist_a:
			return [0, 1]
		else:
			return [0.5, 0.5]


class SumGiverv3:
	def query_pair(self, ret_a, ret_b):
		rew_a = ret_a.sum()
		rew_b = ret_b.sum()

		if rew_a > rew_b:
			return [1, 0]
		elif rew_b > rew_a:
			return [0, 1]
		else:
			return [0.5, 0.5]

class SumProportionalGiverv3:
	def query_pair(self, ret_a, ret_b):
		if check_pareto_dom(ret_a, ret_b):
			return [1, 0]
		elif check_pareto_dom(ret_b, ret_a):
			return [0, 1]
		else:
			rew_a = ret_a.sum()
			rew_b = ret_b.sum()
			total = rew_a + rew_b

			return [rew_a/total, rew_b/total]

class EthicalSumGiverv3:
	def query_pair(self, ret_a, ret_b):
		if check_pareto_dom(ret_a, ret_b):
			return [1, 0]
		elif check_pareto_dom(ret_b, ret_a):
			return [0, 1]
		else :
			rew_eth_a = ret_a[1:].sum()
			rew_eth_b = ret_b[1:].sum()

			if rew_eth_a > rew_eth_b:
				return [1, 0]
			elif rew_eth_b > rew_eth_a:
				return [0, 1]
			else:
				if ret_a[0] > ret_b[0]:
					return [1, 0]
				elif ret_b[0] > ret_a[0]:
					return [0, 1]
				else :
					return [0.5, 0.5]

class EthicalSumThresholdGiverv3:
	def __init__(self, eps_eth, eps):
		self.eps_eth = eps_eth
		self.eps = eps

	def query_pair(self, ret_a, ret_b):
		if check_pareto_dom(ret_a, ret_b):
			return [1, 0]
		elif check_pareto_dom(ret_b, ret_a):
			return [0, 1]
		else :
			rew_eth_a = ret_a[1:].sum()
			rew_eth_b = ret_b[1:].sum()

			if rew_eth_a + self.eps_eth >= rew_eth_b :
				return [1, 0]
			elif rew_eth_b + self.eps_eth >= rew_eth_a:
				return [0, 1]
			else:
				if ret_a[0] + self.eps >= ret_b[0]:
					return [1, 0]
				elif ret_b[0] + self.eps >= ret_a[0]:
					return [0, 1]
				else :
					return [0.5, 0.5]

class ParetoGiverv3:
	def query_pair(self, ret_a, ret_b):
		if check_pareto_dom(ret_a, ret_b):
			return [1, 0]
		elif check_pareto_dom(ret_b, ret_a):
			return [0, 1]
		else:
			return [0.5, 0.5]

class ParetoSoftmaxGiverv3:
	def query_pair(self, ret_a, ret_b):
		if check_pareto_dom(ret_a, ret_b):
			return [1, 0]
		elif check_pareto_dom(ret_b, ret_a):
			return [0, 1]
		else:
			delta_a = ret_a - ret_b
			delta_b = ret_b - ret_a
			delta_norm_a = delta_a/abs(delta_a).sum()
			delta_norm_b = delta_b/abs(delta_b).sum()
			e_a = np.array(np.concatenate( ([delta_norm_a[0]], np.exp(delta_norm_a[1:]))))
			e_b = np.array(np.concatenate( ([delta_norm_b[0]], np.exp(delta_norm_b[1:]))))
			sum_e_d_a = e_a.sum()
			sum_e_d_b = e_b.sum()
			log_s_a = np.log(sum_e_d_a)
			log_s_b = np.log(sum_e_d_b)
			print("delta_a = ",delta_a)
			print("delta_b = ",delta_b)
			print("delta_norm_a = ",delta_norm_a)
			print("delta_norm_b = ",delta_norm_b)
			print("e_a = ",e_a)
			print("e_b = ",e_b)
			print("sum_e_d_a = ",sum_e_d_a)
			print("sum_e_d_b = ",sum_e_d_b)
			print("log_s_a = ",log_s_a)
			print("log_s_b = ",log_s_b)

			# delta_a = ret_a - ret_b
			# delta_b = ret_b - ret_a
			# delta_norm_a = delta_a/abs(delta_a).sum()
			# delta_norm_b = delta_b/abs(delta_b).sum()
			# e_a = np.array(np.concatenate( ([delta_norm_a[0]], np.exp(delta_a[1:]))))
			# e_b = np.array(np.concatenate( ([delta_norm_b[0]], np.exp(delta_b[1:]))))
			# sum_e_d_a = e_a.sum()
			# sum_e_d_b = e_b.sum()
			# log_s_a = np.log(sum_e_d_a)
			# log_s_b = np.log(sum_e_d_b)
			# print("delta_a = ",delta_a)
			# print("delta_b = ",delta_b)
			# print("delta_norm_a = ",delta_norm_a)
			# print("delta_norm_b = ",delta_norm_b)
			# print("e_a = ",e_a)
			# print("e_b = ",e_b)
			# print("sum_e_d_a = ",sum_e_d_a)
			# print("sum_e_d_b = ",sum_e_d_b)
			# print("log_s_a = ",log_s_a)
			# print("log_s_b = ",log_s_b)

			# delta_a = ret_a - ret_b
			# delta_b = ret_b - ret_a
			# delta_norm_a = delta_a/abs(delta_a).sum()
			# delta_norm_b = delta_b/abs(delta_b).sum()
			# e_a = np.exp(delta_norm_a)
			# e_b = np.exp(delta_norm_b)
			# sum_e_d_a = e_a.sum()
			# sum_e_d_b = e_b.sum()
			# log_s_a = np.log(sum_e_d_a)
			# log_s_b = np.log(sum_e_d_b)
			# print("delta_a = ",delta_a)
			# print("delta_b = ",delta_b)
			# print("delta_norm_a = ",delta_norm_a)
			# print("delta_norm_b = ",delta_norm_b)
			# print("e_a = ",e_a)
			# print("e_b = ",e_b)
			# print("sum_e_d_a = ",sum_e_d_a)
			# print("sum_e_d_b = ",sum_e_d_b)
			# print("log_s_a = ",log_s_a)
			# print("log_s_b = ",log_s_b)
			

			if log_s_a > log_s_b:
				return [1, 0]
			elif log_s_b > log_s_a:
				return [0, 1]
			else :
				return [0.5, 0.5]

class EthicalParetoGiverv3:
	def query_pair(self, ret_a, ret_b):
		if check_pareto_dom(ret_a[1:], ret_b[1:]):
			return [1, 0]
		elif check_pareto_dom(ret_b[1:], ret_a[1:]):
			return [0, 1]
		else:
			if ret_a[0] > ret_b[0]:
				return [1, 0]
			elif ret_b[0] > ret_a[0]:
				return [0, 1]
			else :
				return [0.5, 0.5]

class EthicalParetoTestGiverv3:
	def query_pair(self, ret_a, ret_b):
		if check_pareto_dom(ret_a, ret_b):
			return [1, 0]
		elif check_pareto_dom(ret_b, ret_a):
			return [0, 1]
		else:
			delta_a = ret_a - ret_b
			delta_b = ret_b - ret_a
			e_a = np.array(np.concatenate( ([delta_a[0]], np.exp(delta_a[1:]))))
			e_b = np.array(np.concatenate( ([delta_b[0]], np.exp(delta_b[1:]))))
			# exp_d_a = np.exp(delta_a)
			# exp_d_b = np.exp(delta_b)
			soft_a = e_a/(e_a+e_b + 1e-10)
			soft_b = e_b/(e_b+e_a + 1e-10)
			sum_e_d_a = soft_a.sum()
			sum_e_d_b = soft_b.sum()
			log_s_a = np.log(sum_e_d_a)
			log_s_b = np.log(sum_e_d_b)

			print("delta_a = ",delta_a)
			print("delta_b = ",delta_b)
			print("e_a = ",e_a)
			print("e_b = ",e_b)
			print("soft_a = ",soft_a)
			print("soft_b = ",soft_b)
			print("sum_e_d_a = ",sum_e_d_a)
			print("sum_e_d_b = ",sum_e_d_b)
			print("log_s_a = ",log_s_a)
			print("log_s_b = ",log_s_b)

			if log_s_a > log_s_b:
				return [1, 0]
			elif log_s_b > log_s_a:
				return [0, 1]
			else :
				return [0.5, 0.5]

class EthicalParetoThresholdGiverv3:
	def __init__(self, eps_eth=1, eps=1):
		self.eps_eth = eps_eth
		self.eps = eps

	def query_pair(self, ret_a, ret_b):
		pareto_dom_a = []
		for i, ret in enumerate(ret_a[1:]):
			pareto_dom_a.append(ret + self.eps_eth >= ret_b[1+i])
		pareto_dom_a = np.array(pareto_dom_a)

		pareto_dom_b = []
		for i, ret in enumerate(ret_b[1:]):
			pareto_dom_b.append(ret + self.eps_eth >= ret_a[1+i])
		pareto_dom_b = np.array(pareto_dom_b)

		print("ret_a = ", ret_a)
		print("ret_b = ", ret_b)
		print("pareto_dom_a = ", pareto_dom_a)
		print("pareto_dom_b = ", pareto_dom_b)

		if pareto_dom_a.all() and not pareto_dom_b.all():
			return [1, 0]
		elif pareto_dom_b.all() and not pareto_dom_a.all():
			return [0, 1]
		else:
			print("else")
			if ret_a[0] > ret_b[0] + self.eps:
				return [1, 0]
			elif ret_b[0] > ret_a[0] + self.eps:
				return [0, 1]
			else :
				return [0.5, 0.5]



class PreferenceGiverv3:
	def __init__(self, ratio, pbrl=False):
		self.ratio = ratio
		self.d = len(ratio)
		self.ratio_normalized = []
		self.pbrl = pbrl

		ratio_sum = sum(ratio)

		for elem in ratio:
			self.ratio_normalized.append(elem/ratio_sum)

	def query_pair(self, ret_a, ret_b):
		# print("query_pair = "+str(ret_a)+" , "+str(ret_b))

		if self.pbrl:
			ret_a_copy = ret_a.copy()[:-1]
			ret_b_copy = ret_b.copy()[:-1]
		else:
			ret_a_copy = ret_a.copy()
			ret_b_copy = ret_b.copy()

		ret_a_normalized = []
		ret_b_normalized = []

		for i in range(self.d):
			# To avoid numerical instabilities in KL
			ret_a_copy[i] += 1e-5
			ret_b_copy[i] += 1e-5

		ret_a_sum = sum(ret_a_copy)
		ret_b_sum = sum(ret_b_copy)

		for i in range(self.d):
			ret_a_normalized.append(ret_a_copy[i]/ret_a_sum)
			ret_b_normalized.append(ret_b_copy[i]/ret_b_sum)

		# scipy.stats.entropy(pk, qk=None, base=None, axis=0) = S = sum(pk * log(pk / qk), axis=axis)
		# print("ret_a_normalized = ", ret_a_normalized)
		# print("ret_b_normalized = ", ret_b_normalized)
		# print("self.ratio_normalized = ", self.ratio_normalized)
		kl_a = st.entropy(ret_a_normalized, self.ratio_normalized)
		kl_b = st.entropy(ret_b_normalized, self.ratio_normalized)
		# print("kl_a = ", kl_a)
		# print("kl_b = ", kl_b)

		if self.pbrl:
			print(kl_a)
			print(kl_b)

			if ret_a[-1] < ret_b[-1]:
				return [0, 1]
			elif ret_a[-1] > ret_b[-1]:
				return [1, 0]
			else:
				if np.isclose(kl_a, kl_b, rtol=1e-5):
					preference = [0.5, 0.5]
				elif kl_a < kl_b:
					preference = [1, 0]
				else:
					preference = [0, 1]
				return preference
		else:
			if kl_a < kl_b:
				preference = 1
			elif kl_b < kl_a:
				preference = -1
			else:
				preference = 1 if np.random.rand() < 0.5 else -1
			return preference


class MaliciousPreferenceGiverv3:
	def __init__(self, bad_idx):
		self.bad_idx = bad_idx

	def query_pair(self, ret_a, ret_b):
		# Assumes negative reward for bad_idx component
		damage_a = -ret_a[self.bad_idx]
		damage_b = -ret_b[self.bad_idx]

		if damage_a > damage_b:
			preference = 1
		elif damage_b > damage_a:
			preference = -1
		else:
			preference = 1 if np.random.rand() < 0.5 else -1

		return preference


class PbRLPreferenceGiverv2:
	def __init__(self):
		return

	@staticmethod
	def query_pair(ret_a, ret_b, primary=False):
		ppl_saved_a = ret_a[1]
		goal_time_a = ret_a[0]
		ppl_saved_b = ret_b[1]
		goal_time_b = ret_b[0]

		if primary:
			if goal_time_a > goal_time_b:
				preference = [1, 0]
			elif goal_time_b > goal_time_a:
				preference = [0, 1]
			else:
				preference = [0.5, 0.5]
		else:
			if ppl_saved_a > ppl_saved_b:
				preference = [1, 0]
			elif ppl_saved_b > ppl_saved_a:
				preference = [0, 1]
			elif goal_time_a > goal_time_b:
				preference = [1, 0]
			elif goal_time_b > goal_time_a:
				preference = [0, 1]
			else:
				preference = [0.5, 0.5]

		return preference


class PbRLSoftPreferenceGiverv2:
	# Soft preferences
	# Values people saved more but only up to threshold
	def __init__(self, threshold):
		self.threshold = threshold

	def query_pair(self, ret_a, ret_b):
		ppl_saved_a = ret_a[1]
		goal_time_a = ret_a[0]
		ppl_saved_b = ret_b[1]
		goal_time_b = ret_b[0]

		if ppl_saved_a < self.threshold and ppl_saved_b < self.threshold:
			preference = PbRLPreferenceGiverv2.query_pair(ret_a, ret_b)
		else:
			if goal_time_a > goal_time_b:
				preference = [1, 0]
			elif goal_time_b > goal_time_a:
				preference = [0, 1]
			else:
				preference = [0.5, 0.5]

		return preference
