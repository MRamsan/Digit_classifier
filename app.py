import streamlit as st
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

@st.cache_data
def load_and_preprocess():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 784) / 255.0
    x_test = x_test.reshape(-1, 784) / 255.0
    return x_train, y_train, x_test, y_test

def main():
    st.title("MNIST Classification with PCA + Logistic Regression")

    x_train, y_train, x_test, y_test = load_and_preprocess()

    # Show label distribution
    labels, counts = np.unique(y_train, return_counts=True)
    st.write("### Training set label distribution")
    st.bar_chart(dict(zip(labels, counts)))

    # PCA components slider
    n_components = st.slider("Number of PCA components", min_value=2, max_value=100, value=50)

    # Build pipeline
    pipeline = Pipeline([
        ('pca', PCA(n_components=n_components)),
        ('logreg', LogisticRegression(max_iter=1000))
    ])

    if st.button("Train model"):
        with st.spinner("Training..."):
            pipeline.fit(x_train, y_train)
            y_pred = pipeline.predict(x_test)

        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)

        st.write("### Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(report)

        # PCA 2D Visualization
        st.write("### MNIST Visualization after PCA (2D)")
        pca_2d = PCA(n_components=2)
        x_train_2d = pca_2d.fit_transform(x_train)

        fig, ax = plt.subplots(figsize=(8,6))
        scatter = ax.scatter(x_train_2d[:,0], x_train_2d[:,1], c=y_train, cmap='tab10', s=1)
        legend1 = ax.legend(*scatter.legend_elements(), title="Digits")
        ax.add_artist(legend1)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("MNIST dataset projected to 2D with PCA")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
