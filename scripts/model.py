from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier


class IrisClassifier:

	@staticmethod
	def compare_models(X_scaled, y):
		C = 5
		alpha = 1 / (C * len(X_scaled))

		linear_svc_clf = LinearSVC(random_state=42)
		linear_svc_clf.fit(X_scaled, y)

		svc_clf = SVC(random_state=42, kernel="linear")
		svc_clf.fit(X_scaled, y)

		sgd_clf = SGDClassifier(loss="hinge", learning_rate="constant", eta0=0.001, alpha=alpha,
                        max_iter=100000, random_state=42)
		sgd_clf.fit(X_scaled, y)

		print("LinearSVC:                   ", linear_svc_clf.intercept_, linear_svc_clf.coef_)
		print("SVC:                         ", svc_clf.intercept_, svc_clf.coef_)
		print("SGDClassifier(alpha={:.5f}):".format(sgd_clf.alpha), sgd_clf.intercept_, sgd_clf.coef_)

		return linear_svc_clf, svc_clf, sgd_clf


