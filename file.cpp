#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;

int main() {
    // Load the dataset
    MatrixXd dataset;
    ifstream file("Salary_Data.csv");
    string line;
    vector<double> X, y;
    while (getline(file, line)) {
        double x_val, y_val;
        sscanf(line.c_str(), "%lf,%lf", &x_val, &y_val);
        X.push_back(x_val);
        y.push_back(y_val);
    }
    dataset.conservativeResize(X.size(), 2);
    for (int i = 0; i < X.size(); ++i) {
        dataset(i, 0) = X[i];
        dataset(i, 1) = y[i];
    }

    // Split the dataset into the Training set and Test set
    int num_samples = dataset.rows();
    int num_train = num_samples * 2 / 3;
    int num_test = num_samples - num_train;
    MatrixXd X_train = dataset.block(0, 0, num_train, 1);
    MatrixXd y_train = dataset.block(0, 1, num_train, 1);
    MatrixXd X_test = dataset.block(num_train, 0, num_test, 1);
    MatrixXd y_test = dataset.block(num_train, 1, num_test, 1);

    // Define the linear regression model
    MatrixXd X_ones = MatrixXd::Ones(num_train, 1);
    MatrixXd X_train_augmented(num_train, 2);
    X_train_augmented << X_train, X_ones;
    MatrixXd W = (X_train_augmented.transpose() * X_train_augmented).inverse() * X_train_augmented.transpose() * y_train;

    // Predictions
    MatrixXd X_test_ones = MatrixXd::Ones(num_test, 1);
    MatrixXd X_test_augmented(num_test, 2);
    X_test_augmented << X_test, X_test_ones;
    MatrixXd y_pred = X_test_augmented * W;

    // Calculate Mean Squared Error
    double mse = (y_test - y_pred).array().square().mean();

    // Visualize the Training set results
    Mat image(600, 600, CV_8UC3, Scalar(255, 255, 255));
    for (int i = 0; i < num_train; ++i) {
        int x = (X_train(i, 0) - X_train.minCoeff()) / (X_train.maxCoeff() - X_train.minCoeff()) * 600;
        int y = (y_train(i, 0) - y_train.minCoeff()) / (y_train.maxCoeff() - y_train.minCoeff()) * 600;
        circle(image, Point(x, 600 - y), 5, Scalar(0, 0, 255), -1);
    }

    // Visualize the Test set results
    Mat image_test(600, 600, CV_8UC3, Scalar(255, 255, 255));
    for (int i = 0; i < num_test; ++i) {
        int x = (X_test(i, 0) - X_train.minCoeff()) / (X_train.maxCoeff() - X_train.minCoeff()) * 600;
        int y = (y_test(i, 0) - y_train.minCoeff()) / (y_train.maxCoeff() - y_train.minCoeff()) * 600;
        circle(image_test, Point(x, 600 - y), 5, Scalar(0, 0, 255), -1);
    }

    // Show the visualizations
    imshow("Training Set", image);
    imshow("Test Set", image_test);
    waitKey(0);

    cout << "Mean Squared Error: " << mse << endl;

    return 0;
}
