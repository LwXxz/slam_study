#include <iostream>
#include <ceres/ceres.h>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace std;

// 代价函数类
struct CURVE_FITTING_COST
{
    // 冒号后面跟的是赋值，这种写法是C++的特性
    CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

    template <typename T> // 模型参数，有3维
    // 当重载 () 时，您不是创造了一种新的调用函数的方式，相反地，这是创建一个可以传递任意数目参数的运算符函数。
    // 函数后加 const表示函数不可以修改class的成员 只读函数 
    // 先是参数块，然后是残差块
    bool operator()(const T *const abc, T *residual) const
    {
        residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
        return true;
    }

    const int _x, _y;
};

int main(int argc, char **argv)
{
    double ar = 1.0, br = 2.0, cr = 1.0;  // 真实参数值
    double ae = 2.0, be = -1.0, ce = 5.0; // 估计参数值
    int N = 100;                          // 数据点
    double w_sigma = 1.0;                 // 噪声Sigma值
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng; // OpenCV随机数产生器

    // 准备数据
    vector<double> x_data, y_data;
    for (int i = 0; i < N; i++)
    {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma));
    }

    double abc[3] = {ae, be, ce};

    // 构建最小二乘问题
    ceres::Problem problem;
    for (int i = 0; i < N; i++)
    {
        // 三个参数： 
        // 顾名思义主要用于向 Problem 类传递残差模块的信息
        problem.AddResidualBlock(   
            // 第1个模板参数是仿函数CostFunctor、第2个模板参数是残差块中残差的数量、第3个模板参数是第一个参数块中参数的数量，一个abc中有三个参数
            // 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致 不同的求导方式数值求导与自动求导
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(new CURVE_FITTING_COST(x_data[i], x_data[i])),  
            nullptr, // 核函数也叫损失函数
            abc);    // 需要修正的值和优化的模块
    }

    // 配置求解器
    ceres::Solver::Options option;
    option.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY; // 使用CHOLESKY分解来求解 https://blog.csdn.net/xbinworld/article/details/104663481
    option.minimizer_progress_to_stdout = true;  // 输出到cout
    
    ceres::Solver::Summary summary;  // 优化信息

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(option, &problem, &summary);  // 这里就是使用Solve了，不是求解器了, 相当于开始训练的意思？
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double>  timeUsed = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "time spend " << timeUsed.count() << endl;

    // 输出结果
    cout << summary.BriefReport() << endl;
    cout << "estimate a, b, c" << endl;
    for (auto &i : abc)
    {
        cout << i << " ";
    }
    return 0;
}
