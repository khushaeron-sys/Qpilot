from sklearn.datasets import load_breast_cancer, load_iris
load_breast_cancer(as_frame=True).frame.to_csv("data/breast_cancer.csv", index=False)   # target column: "target"
load_iris(as_frame=True).frame.to_csv("data/iris.csv", index=False)
print("wrote data/breast_cancer.csv, data/iris.csv")
