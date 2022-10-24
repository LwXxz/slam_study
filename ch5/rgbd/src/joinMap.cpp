#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp> // for formating strings
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>

using namespace std;
typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

void showPointCloud(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud);

int main(int argc, char **argv)
{
    vector<cv::Mat> colorImgs, depthImgs;
    TrajectoryType poses;

    // 文件输入流
    ifstream fin("/home/cc/CPP_Project/SLAM/ch5/rgbd/data/pose.txt");
    if (!fin)
    {
        cout << "指向有pose.txt的位置" << endl;
        return 1;
    }

    for (int i = 0; i < 5; i++)
    {

        boost::format fmt("../%s/%d.%s"); //图像文件格式 （字符串/数字/字符串   ）
        // 成员函数str()来返回已经格式化好的字符串，如果没有得到按照规则的格式化数据则会抛出异常 类似于python的format %=/
        // cout << (fmt % "home/cc/CPP_Project/SLAM/ch5/rgbd/color" % (i + 1) % "png").str() << endl;
        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "pgm").str(), -1));
        double data[7] = {0};
        // cout << &data[0] << "  "<< &data[1] << endl; // 输出地址，double占8位， 每个元素都是0
        for (auto &d : data) // 把txt文件每行读入，每次读7个，每个元素传进一个值，读五次
        {
            fin >> d; // 从文件读取信息
            // cout << d << endl;
        }
        // cout << data[0] << data[1] << endl; // 输出的是txt中每7个的第一个
        // 四元数中 i + j + k + x
        // 根据旋转矩阵或四元数和平移向量(Vector3d)构造SE(3)矩阵
        Sophus::SE3d pose(Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                          Eigen::Vector3d(data[0], data[1], data[2]));
        // cout << pose.matrix() << endl; // 输出矩阵
        poses.push_back(pose);
    }

    // 计算点云并拼接
    // 相机内参
    double cx = 325.5;          // x方向上的原点平移量
    double cy = 253.5;          // y方向上的原点平移量
    double fx = 518.0;          //焦距
    double fy = 519.0;          //焦距
    double depthScale = 1000.0; //现实世界中1米在深度图中存储为一个depthScale值
    // 生成点云
    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud;
    pointcloud.reserve(1000000); // reserve()函数用来给vector预分配存储区大小，但不对该段内存进行初始化

    for (int i = 0; i < 5; i++)
    {
        cout << "转换图像中" << endl;
        // 读取每一幅图片和其对应的景深 以及对应的SE(3)
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Sophus::SE3d T = poses[i]; // 用SE3d表示的从当前相机坐标系到世界坐标系的变换
        // 遍历整张图片
        for (int v = 0; v < color.rows; v++)
        {
            for (int u = 0; u < color.cols; u++)
            {
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // 深度值
                if (d == 0)
                    continue;
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale; //真实世界中的深度值
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point;

                Vector6d p;               //前三维表示点云的位置，后三维表示点云的颜色
                p.head<3>() = pointWorld; // head<n>()函数是对于Eigen库中的向量类型而言的，表示提取前n个元素
                // opencv中图像的data数组表示把其颜色信息按行优先的方式展成的一维数组！
                // color.step等价于color.cols
                // color.channels()表示图像的通道数
                p[5] = color.data[v * color.step + u * color.channels()];     // blue
                p[4] = color.data[v * color.step + u * color.channels() + 1]; // green
                p[3] = color.data[v * color.step + u * color.channels() + 2]; // red
                pointcloud.push_back(p);
            }
        }

        cout << "点云共有" << pointcloud.size() << "个点." << endl; // 点云共有209236个点.
        showPointCloud(pointcloud);
        return 0;
    }
}

void showPointCloud(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud)
{

    if (pointcloud.empty())
    {
        cerr << "Point cloud is empty!" << endl;
        return;
    }
    // 窗口大小
    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));

    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p : pointcloud)
        {
            glColor3d(p[3] / 255.0, p[4] / 255.0, p[5] / 255.0);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000); // sleep 5 ms
    }
    return;
}
