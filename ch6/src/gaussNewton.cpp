#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>

using namespace std;
using namespace Eigen;

int main(int argc, char **argv){
    cv::RNG rng;
    double ar = 1.0, br = 2.0, cr = 1.0;         // 真实参数值
    double ae = 2.0, be = -1.0, ce = 5.0;        // 估计参数值
    int N = 100;                                 // 数据点
    double w_sigma = 1.0;                        // 噪声Sigma值
    double inv_sigma = 1.0 / w_sigma;

    vector<double> x_data, y_data; // 数据容器和制造数据
    for (int i = 0; i < N; i++){
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma)); // 产生一个高斯分布的随机数 方差为w_sigma
    }

    // 开始Gauss-Newton迭代
    int iteration = 100;  // 迭代次数
    double cost = 0, lastcost = 0;

    // 计时
    // 利用雅可比矩阵求H矩阵
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for (int iter = 0; iter < iteration; iter++){
        Matrix3d H = Matrix3d::Zero();  // 初始化H矩阵
        Vector3d b = Vector3d::Zero();  // 初始化bias
        cost = 0;

        // 雅可比矩阵即一阶导 
        for (int i = 0; i < N; i++)
        {
            double xi = x_data[i], yi = y_data[i];
            double error = yi - (exp(ae * xi * xi + be * xi + ce));
            Vector3d J; // 雅可比矩阵
            J[0] =  -xi * xi * exp(ae * xi * xi + be * xi + ce); // 对ae求导
            J[1] =  -xi * exp(ae * xi * xi + be * xi + ce);  // 对be求导
            J[2] =  -exp(ae * xi * xi + be * xi + ce);  // 对ce求导

            H += inv_sigma * inv_sigma * J * J.transpose(); // 3x3
            b += -inv_sigma * inv_sigma * error * J; // 3 x 1

            cost += error * error;
        }
        // 一种矩阵分解的方式，LDL^T 来求解Hx=b
        Vector3d dx = H.ldlt().solve(b);
        if (isnan(dx[0])){
            cout << "H is nan" << endl;
        }

        if (iter > 0 && cost >= lastcost) {
            cout << "cost: " << cost << ">= last cost: " << lastcost << ", break." << endl;
            break;
        }
        //更新优化变量ae，be和ce！
        ae += dx[0];
        be += dx[1];
        ce += dx[2];
        
        lastcost = cost;

        cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() << "\t\testimated params: " << ae << "," << be << "," << ce << endl;
    }

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    // 计时的一些套路
    chrono::duration<double> washTime = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << washTime.count() << " seconds. " << endl;

    cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;
    return 0;
}