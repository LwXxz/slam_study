#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <unistd.h>

using namespace std;
using namespace Eigen;

string path1 = "./data/left.png";
string path2 = "./data/rigth.png";

// 在pangolin中画图，已写好，无需调整
void showPointCloud(
    const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud);

int main(int argc, char **argv)
{
    // 内参 缩放倍数和原点的平移 在v轴上缩放fx倍，移动cx的长度。 在u轴上缩放fy倍，移动cy的长度
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    // 基线 双目
    double b = 0.573;

    // 读取图像
    cv::Mat left = cv::imread(path1, 0);
    cv::Mat right = cv::imread(path2, 0);
    // SGBM: 立体匹配算法
    cv::Ptr<cv::StereoBM> sgdm = cv::StereoSGBM::create(
        0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);
    cv::Mat disparity_sgbm, disparity;
    sgdm->compute(left, right, disparity_sgbm);
    // 颜色空间转换
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);

    // 生成点云
    vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud;

    for (int u = 0; u < left.rows; u++)
    {
        for (int v = 0; v < left.rows; v++)
        {
            if (disparity.at<float>(u, v) <= 0.0 || disparity.at<float>(u, v) >= 96.0)
                continue;

            Vector4d point(0, 0, 0, left.at<uchar>(u, v) / 255.0); // 前三维为xyz,第四维为颜色

            // 50-56即为p100的公式展示， 已知像素坐标推相机坐标
            // 根据双目模型计算 point 的位置 使用5.14式     
            double x = (u - cx) / fx;
            double y = (v - cx) / fy;
            double depth = fx * b / (disparity.at<float>(u, v)); // at在第u行第v列的像素值，float表示图像像素点的类型 图像到基线的距离

            point[0] = x * depth;
            point[1] = y * depth;
            point[2] = depth;

            pointcloud.push_back(point);
        }
    }
    cv::imshow("disparty", disparity / 96.0);
    cv::waitKey(0);
    // 画出点云
    showPointCloud(pointcloud);
    return 0;   
}

void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    //创建一个pangolin窗口
    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);//根据物体远近，实现遮挡效果
    glEnable(GL_BLEND);//使用颜色混合模型，让物体显示半透明效果
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

   //创建交互视图，显示上一帧图像内容
    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));
    //SetBounds()内的前4个参数分别表示交互视图的大小，均为相对值，范围在0.0至1.0之间
    //第1个参数表示bottom，即为视图最下面在整个窗口中的位置
    //第2个参数为top，即为视图最上面在整个窗口中的位置
    //第3个参数为left，即视图最左边在整个窗口中的位置
    //第4个参数为right，即为视图最右边在整个窗口中的位置
    //第5个参数为aspect，表示横纵比

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);//清空颜色和深度缓存，使得前后帧不会互相干扰
 
        d_cam.Activate(s_cam);//激活显示，并设置相机状态
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);//设置背景颜色为白色
 
        glPointSize(2);
        glBegin(GL_POINTS); //绘制点云
        for (auto &p: pointcloud) {
            glColor3f(p[3], p[3], p[3]); //设置颜色信息
            glVertex3d(p[0], p[1], p[2]);//设置位置信息
        }
        glEnd();
        pangolin::FinishFrame();//按照上面的设置执行渲染
        usleep(5000);   // sleep 5 ms 停止执行5毫秒
    }
    return;
}