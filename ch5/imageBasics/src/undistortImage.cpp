#include <string>   
#include <opencv2/opencv.hpp>

using namespace std;

string image_file = "../img/distorted.png";

int main(int argc, char **argv)
{
    // 本程序实现去畸变部分的代码。尽管我们可以调用OpenCV的去畸变，但自己实现一遍有助于理解
    // 畸变参数
    double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
    // 内参
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

    cv::Mat img = cv::imread(image_file, 0);
    int row = img.rows, colum = img.cols;
    cv::Mat image_undistort = cv::Mat(row, colum, CV_8UC1);

    // 像素坐标系中 原点在左上角 y轴向下 x轴向右 表示坐标则通过矩阵 [y`, x`]来表示 下面是径向和纵向的修改
    for (int v = 0; v < row; v++)
    {
        for (int u = 0; u < colum; u++)
        {
            double x = (u - cx) / fx, y = (v - cy) / fy;
            double r = sqrt(x * x + y * y);
            double x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
            double y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
            double u_distorted = fx * x_distorted + cx;
            double v_distorted = fy * y_distorted + cy;

            // 赋值
            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < colum && v_distorted < row)
            {
                image_undistort.at<uchar>(v, u) = img.at<uchar>((int)v_distorted, (int)u_distorted);
            }
            else
            {
                image_undistort.at<uchar>(v, u) = 0;
            }
        }
    }

    cv::imshow("distorted", img);
    cv::imshow("undistorted", image_undistort);
    cv::waitKey(0);
    return 0;
}