import numpy as np
import math
import scipy.stats as st
import time
import wandb

class PreferenceLearner:
	def __init__(self, n_iter, warmup, d, temperature=None,  cov_range=1):
		self.n_iter = n_iter
		self.warmup = warmup
		self.d = d
		self.accept_rates = None
		self.deltas = []
		self.prefs = []
		self.returns = []

		self.cpt_pior = 0
		self.cpt_nb_steps = 0
		self.cpt_new_acc = 0
		self.cpt_prob_supp = 0
		self.cpt_prior_and_accepted = 0
		self.cpt_prior_new_w = 0

		self.temperature = temperature
		self.cov_range = cov_range

	def log_returns(self, ret_a, ret_b):
		self.returns.append([ret_a, ret_b])

	def log_preference(self, delta, pref):
		self.deltas.append(delta)
		self.prefs.append(pref)

	def pop_returns(self):
		self.returns.pop()

	def pop_preference(self):
		self.deltas.pop()
		self.prefs.pop()


	def w_prior(self, w):
		self.cpt_nb_steps += 1
		if np.linalg.norm(w) <=1 and np.all(np.array(w) >= 0):
			# print("PIOR, w = ", w)
			self.cpt_pior += 1
			return (2**self.d)/(math.pi**(self.d/2)/math.gamma(self.d/2 + 1))
		else:
			return 0

	def w_prior_marius(self, w):
		self.cpt_nb_steps += 1
		if np.linalg.norm(w) <=1 and np.all(np.array(w) >= 0):
			self.cpt_pior += 1
			return 1e100
		else:
			return 0

	def outside_prior_space(self, w):
		return not (np.linalg.norm(w) <=1 and np.all(np.array(w) >= 0))


	def sample_w_prior(self, n):
		sample = np.random.rand(n, self.d)
		w_out = []
		for w in sample:
			w_out.append(list(w/np.linalg.norm(w)))
		return np.array(w_out)

	@staticmethod
	def f_loglik(w, delta, pref):
		# print("w = ", w)
		# print("delta = ", delta)
		# print("pref*np.dot(w, delta) = ", (pref*np.dot(w, delta)))
		# print("np.exp(pref*np.dot(w, delta)) = ", np.exp(pref*np.dot(w, delta)))
		# print("loglik = ", np.log(np.minimum(1, np.exp(pref*np.dot(w, delta)) + 1e-5)))
		return np.log(np.minimum(1, np.exp(pref*np.dot(w, delta)) + 1e-5))

	@staticmethod
	def vanilla_loglik(w, delta, pref):
		return np.log(1/(1+np.exp(-pref*np.dot(w, delta))))

	@staticmethod
	def basic_loglik(w, ret_a, ret_b):
		e_a = np.exp(np.dot(w, ret_a))
		e_b = np.exp(np.dot(w, ret_b))
		return np.log(e_a/(e_a + e_b))

	@staticmethod
	def basic_loglik_temperature(w, ret_a, ret_b, t):
		e_a = np.exp(t*np.dot(w, ret_a))
		e_b = np.exp(t*np.dot(w, ret_b))
		return np.log(e_a/(e_a + e_b))

	@staticmethod
	def propose_w_prob(w1, w2, cov_range):
		q = st.multivariate_normal(mean=w1, cov=cov_range).pdf(w2)
		return q

	@staticmethod
	def propose_w(w_curr, cov_range):
		w_new = st.multivariate_normal(mean=w_curr, cov=cov_range).rvs()
		return w_new

	@staticmethod
	def propose_w_np(w_curr, cov_range):
		w_new = np.random.multivariate_normal(w_curr, np.ones((len(w_curr), len(w_curr)))*cov_range)
		return w_new



	@staticmethod
	def propose_w_normalized(w_curr, cov_range):
		w_new = st.multivariate_normal(mean=w_curr, cov=cov_range).rvs()
		w_new = w_new / np.sum(w_new)
		return w_new

	@staticmethod
	def propose_w_normalized_linalg(w_curr, cov_range):
		w_new = st.multivariate_normal(mean=w_curr, cov=cov_range).rvs()
		w_new = w_new / np.linalg.norm(w_new)
		return w_new

	@staticmethod
	def propose_w_normalized_linalg_positive(w_curr, cov_range):
		w_new = st.multivariate_normal(mean=w_curr, cov=cov_range).rvs()
		w_new = abs(w_new)
		w_new = w_new / np.linalg.norm(w_new)
		return w_new
		
		

	def posterior_log_prob_vanilla(self, deltas, prefs, w):
		f_logliks = []
		for i in range(len(prefs)):
				f_logliks.append(self.vanilla_loglik(w, deltas[i], prefs[i]))
		loglik = np.sum(f_logliks)
		log_prior = np.log(self.w_prior(w) + 1e-5)
		return loglik + log_prior, loglik, log_prior


	def posterior_log_prob_basic_log_lik(self, deltas, prefs, w, returns):
		f_logliks = []
		for i in range(len(prefs)):
			if prefs[i] == 1 :
				f_logliks.append(self.basic_loglik(w, returns[i][0], returns[i][1]))
			else :
				f_logliks.append(self.basic_loglik(w, returns[i][1], returns[i][0]))
			# print("w = ", w)
			# print("ret_a = ", returns[i][0])
			# print("ret_b = ", returns[i][1])
			# print("deltas = ", deltas[i])
			# print("prefs = ", prefs[i])
			# print("loglik 2.0 delta = ", f_logliks[-1])
			# print("loglik vanilla delta = ", self.vanilla_loglik(w, deltas[i], prefs[i]))
			# print("loglik basic a>b = ", self.basic_loglik(w, returns[i][0], returns[i][1]))
			# print("loglik basic b>a = ", self.basic_loglik(w, returns[i][1], returns[i][0]))
		loglik = np.sum(f_logliks)
		log_prior = np.log(self.w_prior(w) + 1e-5)
		# print("sum loglik = ", loglik)
		# print("prior = ", log_prior)

		return loglik + log_prior, loglik, log_prior

	def posterior_log_prob_basic_log_lik_temperature(self, deltas, prefs, w, returns, t):
		f_logliks = []
		for i in range(len(prefs)):
			if prefs[i] == 1 :
				f_logliks.append(self.basic_loglik_temperature(w, returns[i][0], returns[i][1], t))
			else :
				f_logliks.append(self.basic_loglik_temperature(w, returns[i][1], returns[i][0], t))
		loglik = np.sum(f_logliks)
		# log_prior = np.log(self.w_prior(w) + 1e-5)
		log_prior = np.log(self.w_prior_marius(w) + 1e-5)

		return loglik + log_prior, loglik, log_prior

	def posterior_log_prob_print(self, deltas, prefs, w, returns, t):
		f_logliks = []
		f_logliks_basic = []
		f_logliks_temperature = []
		for i in range(len(prefs)):
			if prefs[i] == 1 :
				f_logliks.append(self.f_loglik(w, deltas[i], prefs[i]))
				f_logliks_basic.append(self.basic_loglik(w, returns[i][0], returns[i][1]))
				f_logliks_temperature.append(self.basic_loglik_temperature(w, returns[i][0], returns[i][1], t))
			else :
				f_logliks.append(self.f_loglik(w, deltas[i], prefs[i]))
				f_logliks_basic.append(self.basic_loglik(w, returns[i][0], returns[i][1]))
				f_logliks_temperature.append(self.basic_loglik_temperature(w, returns[i][1], returns[i][0], t))
		loglik = np.sum(f_logliks)
		logliks_basic = np.sum(f_logliks_basic)
		logliks_temperature = np.sum(f_logliks_temperature)
		log_prior = np.log(self.w_prior(w) + 1e-5)
		# print("loglik = ", loglik)
		# print("logliks_basic = ", logliks_basic)
		# print("logliks_temperature = ", logliks_temperature)
		# print("log_prior = ", log_prior)
		return loglik + log_prior, logliks_basic + log_prior, logliks_temperature + log_prior, loglik, logliks_basic, logliks_temperature, log_prior

	def posterior_log_prob(self, deltas, prefs, w):
		f_logliks = []
		for i in range(len(prefs)):
			f_logliks.append(self.f_loglik(w, deltas[i], prefs[i]))
		loglik = np.sum(f_logliks)
		log_prior = np.log(self.w_prior(w) + 1e-5)

		return loglik + log_prior, loglik, log_prior

	def mcmc_vanilla(self, w_init='mode'):
		if w_init == 'mode':
			w_init = [0 for i in range(self.d)]

		w_arr = []
		w_curr = w_init
		accept_rates = []
		accept_cum = 0

		for i in range(1, self.warmup + self.n_iter + 1):
			w_new = self.propose_w(w_curr, self.cov_range)

			prob_curr, loglik_curr, log_prior_curr = self.posterior_log_prob(self.deltas, self.prefs, w_curr)
			prob_new, loglik_new, log_prior_new = self.posterior_log_prob(self.deltas, self.prefs, w_new)

			if prob_new > prob_curr:
				acceptance_ratio = 1
			else:
				qr = self.propose_w_prob(w_curr, w_new, self.cov_range) / self.propose_w_prob(w_new, w_curr, self.cov_range)
				acceptance_ratio = np.exp(prob_new - prob_curr) * qr
			acceptance_prob = min(1, acceptance_ratio)

			if acceptance_prob > st.uniform(0, 1).rvs():
				w_curr = w_new
				accept_cum = accept_cum + 1
				w_arr.append(w_new)
			else:
				w_arr.append(w_curr)

			accept_rates.append(accept_cum / i)

		self.accept_rates = np.array(accept_rates)[self.warmup:]

		return np.array(w_arr)[self.warmup:]

	def mcmc_test(self, w_init='mode', prop_w_mode="moral", posterior_mode="moral", step=None):
		if w_init == 'mode':
			w_init = [0 for i in range(self.d)]

		w_arr = []
		w_curr = w_init
		accept_rates = []
		accept_cum = 0

		# print("posterior_prob w_init = ", self.posterior_log_prob_test_prints(self.deltas, self.prefs, w_init, self.returns))

		mean_prob_w_new = 0
		mean_prob_w_new_log_lik = 0
		mean_prob_w_new_log_prior = 0

		for i in range(1, self.warmup + self.n_iter + 1):
			w_new = None
			if prop_w_mode == "moral":
				w_new = self.propose_w(w_curr, self.cov_range)
				# w_new = self.propose_w_np(w_curr)
			elif prop_w_mode == "normalized_linalg":
				w_new = self.propose_w_normalized_linalg(w_curr, self.cov_range)
			elif prop_w_mode == "normalized_linalg_positive":
				w_new = self.propose_w_normalized_linalg_positive(w_curr, self.cov_range)
			elif prop_w_mode == "normalized":
				w_new = self.propose_w_normalized(w_curr, self.cov_range)

			

			if posterior_mode == "moral":
				prob_curr, loglik_curr, log_prior_curr = self.posterior_log_prob(self.deltas, self.prefs, w_curr)
				prob_new, loglik_new, log_prior_new = self.posterior_log_prob(self.deltas, self.prefs, w_new)
			elif posterior_mode == "basic" : 
				prob_curr, loglik_curr, log_prior_curr = self.posterior_log_prob_basic_log_lik(self.deltas, self.prefs, w_curr, self.returns)
				prob_new, loglik_new, log_prior_new = self.posterior_log_prob_basic_log_lik(self.deltas, self.prefs, w_new, self.returns)
			elif posterior_mode == "basic_temperature" : 
				prob_curr, loglik_curr, log_prior_curr = self.posterior_log_prob_basic_log_lik_temperature(self.deltas, self.prefs, w_curr, self.returns, self.temperature)
				prob_new, loglik_new, log_prior_new = self.posterior_log_prob_basic_log_lik_temperature(self.deltas, self.prefs, w_new, self.returns, self.temperature)
			# elif posterior_mode == "print" : 
			# 	# print("w_curr = ", w_curr)
			# 	prob_curr_moral, prob_curr, prob_curr_temperature, loglik_moral_curr, logliks_basic_curr, logliks_temperature_curr, log_prior_curr = self.posterior_log_prob_print(self.deltas, self.prefs, w_curr, self.returns, self.temperature)
			# 	# print("w_new = ", w_new)
			# 	prob_new_moral, prob_new, prob_new_temperature, loglik_moral_new, logliks_basic_new, logliks_temperature_new, log_prior_new = self.posterior_log_prob_print(self.deltas, self.prefs, w_new, self.returns, self.temperature)
			elif posterior_mode == "vanilla" : 
				prob_curr, loglik_curr, log_prior_curr = self.posterior_log_prob_vanilla(self.deltas, self.prefs, w_curr)
				prob_new, loglik_new, log_prior_new = self.posterior_log_prob_vanilla(self.deltas, self.prefs, w_curr)

			mean_prob_w_new += prob_new
			mean_prob_w_new_log_lik += loglik_new
			mean_prob_w_new_log_prior += log_prior_new

			if self.outside_prior_space(w_new):
				self.cpt_prior_new_w += 1

			if prob_new > prob_curr:
				self.cpt_prob_supp += 1
				acceptance_ratio = 1
			else:
				qr_a = self.propose_w_prob(w_curr, w_new, self.cov_range)
				qr_b = self.propose_w_prob(w_new, w_curr, self.cov_range)
				qr = qr_a / qr_b

				acceptance_ratio = np.exp(prob_new - prob_curr) * qr
				
			acceptance_prob = min(1, acceptance_ratio)

			if acceptance_prob > st.uniform(0, 1).rvs():
				w_curr = w_new
				# prob_curr = prob_new ?
				accept_cum = accept_cum + 1
				w_arr.append(w_new)
				self.cpt_new_acc += 1
				if self.outside_prior_space(w_new):
					self.cpt_prior_and_accepted += 1
			else:
				w_arr.append(w_curr)

			accept_rates.append(accept_cum / i)

		self.accept_rates = np.array(accept_rates)[self.warmup:]

		nb_steps = self.warmup + self.n_iter
		# print("self.cpt_nb_steps = ", self.cpt_nb_steps)
		# print("self.cpt_prob_supp = ", self.cpt_prob_supp)
		# print("self.cpt_new_acc = ", self.cpt_new_acc)
		# print("self.cpt_pior = ",self.cpt_pior)
		print("nb new w hit by prior = " + str(self.cpt_prior_new_w) + " / " + str(nb_steps))
		print("nb accepted = ", str(self.cpt_prior_and_accepted)+ " / " + str(self.cpt_prior_new_w))
		print("nb new w that weren't hit by prior = " + str(nb_steps - self.cpt_prior_new_w) + " / " + str(nb_steps) + "("  + str(round(100*(nb_steps - self.cpt_prior_new_w)/nb_steps,1)) + "%)")
		# print("pourcentage of new w in the prior space = " + str(round(100*(nb_steps - self.cpt_prior_new_w)/nb_steps,1)))
		print("nb new w accepted = " + str(self.cpt_new_acc) + " / " + str(nb_steps))
		print("nb new w prob sup to curr w = " + str(self.cpt_prob_supp) + " / " + str(nb_steps))

		mean_prob_w_new = mean_prob_w_new / nb_steps
		mean_prob_w_new_log_lik = mean_prob_w_new_log_lik / nb_steps
		mean_prob_w_new_log_prior = mean_prob_w_new_log_prior / nb_steps

		if step != None:
			print("step")
			wandb.log({"nb new w outside prior space": self.cpt_prior_new_w}, step=step)
			wandb.log({"nb accepted oustide prior space": self.cpt_prior_and_accepted}, step=step)
			wandb.log({"nb new w inside prior space": nb_steps - self.cpt_prior_new_w}, step=step)
			wandb.log({"nb accepted": self.cpt_new_acc}, step=step)
			wandb.log({"nb accepted without prob sup": self.cpt_prob_supp}, step=step)
			wandb.log({"mean_prob_w_new": mean_prob_w_new}, step=step)
			wandb.log({"mean_prob_w_new_log_lik": mean_prob_w_new_log_lik}, step=step)
			wandb.log({"mean_prob_w_new_log_prior": mean_prob_w_new_log_prior}, step=step)

		self.cpt_pior = 0
		self.cpt_nb_steps = 0
		self.cpt_prob_supp = 0
		self.cpt_new_acc = 0
		self.cpt_prior_and_accepted = 0
		self.cpt_prior_new_w = 0

		return np.array(w_arr)[self.warmup:]


	def mcmc_print(self, w_init='mode', prop_w_mode="moral", posterior_mode="moral", step = None):
		if w_init == 'mode':
			w_init = [0 for i in range(self.d)]

		w_arr = []
		w_curr = w_init
		accept_rates = []
		accept_cum = 0

		# print("posterior_prob w_init = ", self.posterior_log_prob_test_prints(self.deltas, self.prefs, w_init, self.returns))

		mean_prob_w_new_moral = 0
		mean_prob_w_new_basic = 0
		mean_prob_w_new_temperature = 0

		for i in range(1, self.warmup + self.n_iter + 1):
			w_new = None
			if prop_w_mode == "moral":
				w_new = self.propose_w(w_curr, self.cov_range)
				# w_new = self.propose_w_np(w_curr)
			elif prop_w_mode == "normalized_linalg":
				w_new = self.propose_w_normalized_linalg(w_curr, self.cov_range)
			elif prop_w_mode == "normalized_linalg_positive":
				w_new = self.propose_w_normalized_linalg_positive(w_curr, self.cov_range)
			elif prop_w_mode == "normalized":
				w_new = self.propose_w_normalized(w_curr, self.cov_range)

			

			# print("w_curr = ", w_curr)
			prob_curr_moral, prob_curr_basic, prob_curr_temperature, loglik_moral_curr, logliks_basic_curr, logliks_temperature_curr, log_prior_curr = self.posterior_log_prob_print(self.deltas, self.prefs, w_curr, self.returns, self.temperature)
			# print("w_new = ", w_new)
			prob_new_moral, prob_new_basic, prob_new_temperature, loglik_moral_new, logliks_basic_new, logliks_temperature_new, log_prior_new = self.posterior_log_prob_print(self.deltas, self.prefs, w_new, self.returns, self.temperature)


			mean_prob_w_new_moral += prob_new_moral
			mean_prob_w_new_basic += prob_new_basic
			mean_prob_w_new_temperature += prob_new_temperature


			prob_curr = prob_curr_basic
			prob_new = prob_new_basic

			if self.outside_prior_space(w_new):
				self.cpt_prior_new_w += 1

			if prob_new > prob_curr:
				self.cpt_prob_supp += 1
				acceptance_ratio = 1
			else:
				qr_a = self.propose_w_prob(w_curr, w_new, self.cov_range)
				qr_b = self.propose_w_prob(w_new, w_curr, self.cov_range)
				qr = qr_a / qr_b

				acceptance_ratio = np.exp(prob_new - prob_curr) * qr

				acceptance_ratio_basic = np.exp(prob_new_basic - prob_curr_basic) * qr
				acceptance_ratio_moral = np.exp(prob_new_moral - prob_curr_moral) * qr
				acceptance_ratio_temperature = np.exp(prob_new_temperature - prob_curr_temperature) * qr

				# print("w_curr = ", w_curr)
				# print("w_new = ", w_new)
				# print("prob_curr_moral = ", prob_curr_moral)
				# print("prob_new_moral = ", prob_new_moral)
				# print("prob_curr_basic = ", prob_curr_basic)
				# print("prob_new_basic = ", prob_new_basic)
				# print("prob_curr_temperature = ", prob_curr_temperature)
				# print("prob_new_temperature = ", prob_new_temperature)
				# if prob_new <= prob_curr:
				#     print("qr_a = ", qr_a)
				#     print("qr_b = ", qr_b)
				#     print("qr = ", qr)
				#     print("acceptance_ratio_moral = ", acceptance_ratio_moral)
				#     print("acceptance_ratio_basic = ", acceptance_ratio_basic)
				#     print("acceptance_ratio_temperature = ", acceptance_ratio_temperature)
				
				# print("acceptance_ratio = ", acceptance_ratio)
			acceptance_prob = min(1, acceptance_ratio)

			if acceptance_prob > st.uniform(0, 1).rvs():
				# if prob_new < -9:
					# print("w_curr = ", w_curr)
					# print("w_new = ", w_new)
					# print("prob_curr_moral = ", prob_curr_moral)
					# print("prob_new_moral = ", prob_new_moral)
					# print("prob_curr_basic = ", prob_curr)
					# print("prob_new_basic = ", prob_new)
					# print("prob_curr_temperature = ", prob_curr_temperature)
					# print("prob_new_temperature = ", prob_new_temperature)
					# if prob_new <= prob_curr:
					#     print("qr_a = ", qr_a)
					#     print("qr_b = ", qr_b)
					#     print("qr = ", qr)
					#     print("acceptance_ratio_moral = ", acceptance_ratio_moral)
					#     print("acceptance_ratio_basic = ", acceptance_ratio)
					#     print("acceptance_ratio_temperature = ", acceptance_ratio_temperature)
					# time.sleep(30)
				w_curr = w_new
				# prob_curr = prob_new ?
				accept_cum = accept_cum + 1
				w_arr.append(w_new)
				self.cpt_new_acc += 1
				if self.outside_prior_space(w_new):
					self.cpt_prior_and_accepted += 1
			else:
				w_arr.append(w_curr)

			accept_rates.append(accept_cum / i)

		self.accept_rates = np.array(accept_rates)[self.warmup:]

		nb_steps = self.warmup + self.n_iter
		# print("self.cpt_nb_steps = ", self.cpt_nb_steps)
		# print("self.cpt_prob_supp = ", self.cpt_prob_supp)
		# print("self.cpt_new_acc = ", self.cpt_new_acc)
		# print("self.cpt_pior = ",self.cpt_pior)
		print("nb new w hit by prior = " + str(self.cpt_prior_new_w) + " / " + str(nb_steps))
		print("nb accepted = ", str(self.cpt_prior_and_accepted)+ " / " + str(self.cpt_prior_new_w))
		print("nb new w that weren't hit by prior = " + str(nb_steps - self.cpt_prior_new_w) + " / " + str(nb_steps) + "("  + str(round(100*(nb_steps - self.cpt_prior_new_w)/nb_steps,1)) + "%)")
		# print("pourcentage of new w in the prior space = " + str(round(100*(nb_steps - self.cpt_prior_new_w)/nb_steps,1)))
		print("nb new w accepted = " + str(self.cpt_new_acc) + " / " + str(nb_steps))
		print("nb new w prob sup to curr w = " + str(self.cpt_prob_supp) + " / " + str(nb_steps))
		self.cpt_pior = 0
		self.cpt_nb_steps = 0
		self.cpt_prob_supp = 0
		self.cpt_new_acc = 0
		self.cpt_prior_and_accepted = 0
		self.cpt_prior_new_w = 0

		mean_prob_w_new_moral = mean_prob_w_new_moral/nb_steps
		mean_prob_w_new_basic = mean_prob_w_new_basic/nb_steps
		mean_prob_w_new_temperature = mean_prob_w_new_temperature/nb_steps

		if step != None:
			wandb.log({"nb new w outside prior space": self.cpt_prior_new_w}, step=step)
			wandb.log({"nb accepted oustide prior space": self.cpt_prior_and_accepted}, step=step)
			wandb.log({"nb new w inside prior space": nb_steps - self.cpt_prior_new_w}, step=step)
			wandb.log({"nb accepted": self.cpt_new_acc}, step=step)
			wandb.log({"nb accepted without prob sup": self.cpt_prob_supp}, step=step)
			wandb.log({"mean_prob_w_new_moral": mean_prob_w_new_moral}, step=step)
			wandb.log({"mean_prob_w_new_basic": mean_prob_w_new_basic}, step=step)
			wandb.log({"mean_prob_w_new_temperature": mean_prob_w_new_temperature}, step=step)

		return np.array(w_arr)[self.warmup:]


class VolumeBuffer:
	def __init__(self, dim_ratio, auto_pref=True):
		self.auto_pref = auto_pref
		self.best_volume = -np.inf
		self.best_delta = None
		self.best_observed_returns = None
		self.best_returns = None
		self.observed_logs = []
		self.objective_logs = []
		self.dimension_ratio = dim_ratio
		self.objective_logs_sum = []
		self.observed_logs_sum = []

	def log_statistics(self, statistics):
		# print("objective LOG : ", statistics)
		self.objective_logs.append(statistics)

	def log_rewards(self, rewards):
		# print("observed LOG : ", rewards)
		self.observed_logs.append(rewards)

	def log_rewards_sum(self, observed_logs_sum):
		self.observed_logs_sum = observed_logs_sum

	def log_statistics_sum(self, objective_logs_sum):
		self.objective_logs_sum = objective_logs_sum

	def log_rewards_2(self, observed_logs_sum):
		self.observed_logs_sum = np.concatenate((self.observed_logs_sum, observed_logs_sum))

	def log_statistics_2(self, objective_logs_sum):
		self.objective_logs_sum = np.concatenate((self.objective_logs_sum, objective_logs_sum))

	@staticmethod
	def volume_removal(w_posterior, delta):
		expected_volume_a = 0
		expected_volume_b = 0
		for w in w_posterior:
			# print("w :",w)
			# print("delta :",delta)
			expected_volume_a += (1 - PreferenceLearner.f_loglik(w, delta, 1))
			expected_volume_b += (1 - PreferenceLearner.f_loglik(w, delta, -1))

		# print("expected_volume_a : ", expected_volume_a)
		# print("expected_volume_b : ", expected_volume_b)
		# print("len(w_posterior) = ", len(w_posterior)) # == batch size des rollout ?
		return min(expected_volume_a / len(w_posterior), expected_volume_b / len(w_posterior))

	@staticmethod
	def volume_removal_basic_log_lik(w_posterior, ret_a, ret_b, delta, temperature=1):
		expected_volume_a = 0
		expected_volume_b = 0
		for w in w_posterior:
			expected_volume_a += (1 - PreferenceLearner.basic_loglik_temperature(w, ret_a, ret_b, temperature))
			expected_volume_b += (1 - PreferenceLearner.basic_loglik_temperature(w, ret_b, ret_a, temperature))
		mini = min(expected_volume_a / len(w_posterior), expected_volume_b / len(w_posterior))
		return mini




	def sample_return_pair_no_batch_reset(self):
		rand_idx = np.random.choice(np.arange(len(self.observed_logs_sum)), 2, replace=False)
		new_returns_a = self.observed_logs_sum[rand_idx[0]]
		new_returns_b = self.observed_logs_sum[rand_idx[1]]

		# Also return ground truth logs for automatic preferences
		if self.auto_pref:
			logs_a = self.objective_logs_sum[rand_idx[0]]
			logs_b = self.objective_logs_sum[rand_idx[1]]
			return np.array(new_returns_a), np.array(new_returns_b), logs_a, logs_b
		else:
			return np.array(new_returns_a), np.array(new_returns_b)


	def sample_return_pair_v2(self):
		observed_logs_returns = self.observed_logs_sum
		rand_idx = np.random.choice(np.arange(len(observed_logs_returns)), 2, replace=False)
		new_returns_a = observed_logs_returns[rand_idx[0]]
		new_returns_b = observed_logs_returns[rand_idx[1]]

		# Reset observed logs
		self.observed_logs_sum = []

		# Also return ground truth logs for automatic preferences
		if self.auto_pref:
			objective_logs_returns = self.objective_logs_sum
			logs_a = objective_logs_returns[rand_idx[0]][:self.dimension_ratio]
			logs_b = objective_logs_returns[rand_idx[1]][:self.dimension_ratio]
			self.objective_logs_sum = []
			return np.array(new_returns_a), np.array(new_returns_b), logs_a, logs_b
		else:
			return np.array(new_returns_a), np.array(new_returns_b)


	def sample_return_pair(self):
		# len(self.observed_logs) = 74 donc tous les états de 1 trajectoire, car on a demandé un batchsize de 12 pour 12 workers ?
		# len(self.observed_logs[0]) = 12 donc tous les rewards des 12 workers à l'état 0 ?
		# len(self.observed_logs[0][0]) = 3 donc les rewards des 3 objectifs de l'état 0 pour le workers 0 ?
		# len(observed_logs_returns) = 12 donc on a fait la somme des rewards pour tous les états des trajectoires de chaque worker ?


		# observed_logs_returns est l'ensemble des vectorized_rewards, pour chaque état des trajectoires mémorisées. 
		# vectorized_rewards étant la liste des rewards pour chaque objectif selon chaque agent 
		# (vectorized_rewards[i] = liste des rewards pour l'objectif i, vectorized_rewards[i][j] = reward pour l'objectif i et l'expert j)
		# On fait ici la somme des vectorized_rewards pour tous les états mémorisé ?
		# On aurait donc une liste pour chaque objectif de sommes des rewards (effectifs, et d'experts) pour chaque état ?
		# print("len(self.observed_logs) = ", len(self.observed_logs))
		# print("len(self.observed_logs[0]) = ", len(self.observed_logs[0]))
		# print("len(self.observed_logs[0][0]) = ", len(self.observed_logs[0][0]))
		observed_logs_returns = np.array(self.observed_logs).sum(axis=0)
		# print("len(observed_logs_returns) = ", len(observed_logs_returns))
		# On choisit deux états parmi tous les états des trajec
		rand_idx = np.random.choice(np.arange(len(observed_logs_returns)), 2, replace=False)
		# print("rand_idx = ", rand_idx)

		# v2-Environment comparison
		#new_returns_a = observed_logs_returns[rand_idx[0]]
		#new_returns_b = observed_logs_returns[rand_idx[1]]

		# v3-Environment comparison (vase agnostic)
		# print("len(observed_logs_returns[rand_idx[0]]) = ", len(observed_logs_returns[rand_idx[0]]))
		new_returns_a = observed_logs_returns[rand_idx[0], 0:3]
		new_returns_b = observed_logs_returns[rand_idx[1], 0:3]
		# print("observed_logs_returns = ", observed_logs_returns)
		# print("observed_logs_returns 0 = ", new_returns_a)
		# print("observed_logs_returns 1 = ", new_returns_b)

		# Reset observed logs
		self.observed_logs = []

		# Also return ground truth logs for automatic preferences
		if self.auto_pref:
			objective_logs_returns = np.array(self.objective_logs).sum(axis=0)

			# v2-Environment comparison
			#logs_a = objective_logs_returns[rand_idx[0]]
			#logs_b = objective_logs_returns[rand_idx[1]]

			# v3-Environment comparison (vase agnostic)
			logs_a = objective_logs_returns[rand_idx[0], 0:3]
			logs_b = objective_logs_returns[rand_idx[1], 0:3]
			# print("objective_logs_returns = ", objective_logs_returns)
			# print("logs_a = ", logs_a)
			# print("logs_b = ", logs_b)

			self.objective_logs = []
			return np.array(new_returns_a), np.array(new_returns_b), logs_a, logs_b
		else:
			return np.array(new_returns_a), np.array(new_returns_b)



	def compare_delta(self, w_posterior, new_returns_a, new_returns_b, logs_a=None, logs_b=None, random=False):
		delta = new_returns_a - new_returns_b
		# print("delta = ", delta)
		volume_delta = self.volume_removal(w_posterior, delta)
		# print("volume_delta = ", volume_delta)
		# print("best_volume = ", self.best_volume)
		if volume_delta > self.best_volume or random:
			self.best_volume = volume_delta
			self.best_delta = delta
			self.best_observed_returns = (new_returns_a, new_returns_b)
			# print("self.best_observed_returns ", self.best_observed_returns)
			self.best_returns = (logs_a, logs_b)
			# print("self.best_returns ", self.best_returns)

	def compare_delta_basic_log_lik(self, w_posterior, temperature):
		new_returns_a, new_returns_b, logs_a, logs_b = self.sample_return_pair_no_batch_reset()
		delta = new_returns_a - new_returns_b
		volume_delta = self.volume_removal_basic_log_lik(w_posterior, new_returns_a, new_returns_b, delta, temperature)
		if volume_delta > self.best_volume:
			self.best_volume = volume_delta
			self.best_delta = delta
			self.best_observed_returns = (new_returns_a, new_returns_b)
			self.best_returns = (logs_a, logs_b)

	def compare_MORAL(self, w_posterior):
		new_returns_a, new_returns_b, logs_a, logs_b = self.sample_return_pair_no_batch_reset()
		delta = new_returns_a - new_returns_b
		volume_delta = self.volume_removal(w_posterior, delta)
		if volume_delta > self.best_volume:
			self.best_volume = volume_delta
			self.best_delta = delta
			self.best_observed_returns = (new_returns_a, new_returns_b)
			self.best_returns = (logs_a, logs_b)


	def compare_EUS(self, w_posterior, w_posterior_mean, preference_learner):
		new_returns_a, new_returns_b, logs_a, logs_b = self.sample_return_pair_no_batch_reset()
		delta = new_returns_a - new_returns_b
		# np.array(new_returns_a), np.array(new_returns_b), logs_a, logs_b

		# 1rst mcmc (a>b)
		preference_learner.log_preference(delta, 1)
		preference_learner.log_returns(new_returns_a, new_returns_b)
		w_posterior_sup = preference_learner.mcmc_vanilla(w_posterior_mean)
		# w_posterior = preference_learner.mcmc_test(w_posterior_mean, posterior_mode="moral", prop_w_mode="basic_temperature")
		w_posterior_mean_sup = w_posterior_sup.mean(axis=0)
		w_posterior_mean_sup = w_posterior_mean_sup/np.linalg.norm(w_posterior_mean_sup)
		EUS_sup = preference_learner.posterior_log_prob(preference_learner.deltas, preference_learner.prefs, w_posterior_mean_sup)
		preference_learner.pop_preference()
		preference_learner.pop_returns()

		# 2nd mcmc (a>b)
		preference_learner.log_preference(delta, -1)
		preference_learner.log_returns(new_returns_a, new_returns_b)
		w_posterior_inf = preference_learner.mcmc_vanilla(w_posterior_mean)
		# w_posterior = preference_learner.mcmc_test(w_posterior_mean, posterior_mode="moral", prop_w_mode="basic_temperature")
		w_posterior_mean_inf = w_posterior_inf.mean(axis=0)
		w_posterior_mean_inf = w_posterior_mean_inf/np.linalg.norm(w_posterior_mean_inf)
		
		EUS_inf = preference_learner.posterior_log_prob(preference_learner.deltas, preference_learner.prefs, w_posterior_mean_inf)
		preference_learner.pop_preference()
		preference_learner.pop_returns()

		volume_delta = EUS_sup + EUS_inf
		print(f'rew_a{new_returns_a}')
		print(f'rew_b{new_returns_b}')
		print(f'log_a{logs_a}')
		print(f'log_b{logs_b}')
		print(f'Posterior Mean Sup{w_posterior_mean_sup}')
		print(f'Posterior Mean Inf{w_posterior_mean_inf}')
		print(f'EUS_sup{EUS_sup}')
		print(f'EUS_inf{EUS_inf}')
		print(f'EUS{volume_delta}')

		if volume_delta > self.best_volume:
			self.best_volume = volume_delta
			self.best_delta = delta
			self.best_observed_returns = (new_returns_a, new_returns_b)
			self.best_returns = (logs_a, logs_b)



	def get_best(self):
		ret_a, ret_b = self.best_returns
		rew_a, rew_b = self.best_observed_returns
		return ret_a, ret_b, rew_a, rew_b

	def reset(self):
		self.best_volume = -np.inf
		self.best_delta = None
		self.best_returns = None
		self.best_observed_returns = (None, None)

	def reset_batch(self):
		self.observed_logs = []
		self.objective_logs = []
		self.objective_logs_sum = []
		self.observed_logs_sum = []

	def get_data(self):
		return self.objective_logs, self.observed_logs

