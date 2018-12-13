from scripts.data import IrisData
from scripts.model import IrisClassifier


# Step 1
# getting experimental data
iris_data_obj = IrisData()
X, X_scaled, y, scaler = iris_data_obj.get_experimental_data_set()

# Step 2
# training model
iris_clf = IrisClassifier()
linear_svc_clf, svc_clf, sgd_clf = iris_clf.compare_models(X_scaled, y)

# Step 3
# show visualization
iris_data_obj.show_visualization(X, y, scaler, linear_svc_clf, svc_clf, sgd_clf)
