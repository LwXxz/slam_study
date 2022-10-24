#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cout << "usage: feature_extraction img1 img2" << endl;
        return 1;
    }

    // read image
    Mat img1 = imread(argv[1], IMREAD_COLOR);
    Mat img2 = imread(argv[2], IMREAD_COLOR);
    // 相当于一个 if 语句
    assert(img1.data != nullptr && img2.data != nullptr);

    // initialization
    std::vector<KeyPoint> keypoint1, keypoint2;
    Mat descriptor1, descriptor2;
    // 智能指针Ptr，使用智能指针最方便的就是在我们使用new申请动态内存的时候，不需要用delete释放，系统会自动释放
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming"); // Hamming distance

    // step 1 detecte the Oriented FAST position
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    detector->detect(img1, keypoint1);
    detector->detect(img2, keypoint2);

    // step 2 calculate the BRIEF descriptor 
    descriptor->compute(img1, keypoint1, descriptor1); // 访问迭代器的成员 计算各自图片的描述子
    descriptor->compute(img2, keypoint2, descriptor2);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> timeUsed = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "extract ORB cost: " << timeUsed.count() << 's' << endl;

    Mat outputImg;
    drawKeypoints(img1, keypoint1, outputImg, Scalar::all(-1), DrawMatchesFlags::DEFAULT);  // 颜色随机 Scalar::all(-1) 
    imshow("ORB features", outputImg);

    // step 3 match the BRIEF descriptor
    vector<DMatch> matches;
    t1 = chrono::steady_clock::now();
    matcher->match(descriptor1, descriptor2, matches);
    t2 = chrono::steady_clock::now();
    timeUsed = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "match ORB cost " << timeUsed.count() << endl;

    cout << "keypoint1 size: "<< keypoint1.size() << "keypoint2 size: " << keypoint2.size() << endl; // both are 500
    cout << "keypoint1[0] type: "<< typeid(keypoint1[0]).name() << " " << "keypoint2[0] type: " << " " << typeid(keypoint2[0]).name() << endl; // both are 500
    cout << "keypoint1 pt: " << keypoint1[0].pt.x << " " << "keypoint2 pt: " << keypoint2[0].pt << endl;
    cout << "keypoint1 size: " << keypoint1[0].size << " " << "keypoint2 size: " << keypoint2[0].size << endl;
    cout << "keypoint1 angle: " << keypoint1[0].angle << " " << "keypoint2 angle: " << keypoint2[0].angle << endl;
    cout << "keypoint1 class_id: " << keypoint1[0].class_id << " " << "keypoint2 class_id: " << keypoint2[0].class_id << endl;
    

    // select the match point
    // compute the max distance and the smallest distance
    // 返回范围内的最小和最大元素, 二进制函数(可以省去)，接受范围中的两个元素作为参数，并返回可转换为bool的值   返回指针数组[first， last]为其在容器中的索引
    auto min_max = minmax_element(matches.begin(), matches.end(), [](const DMatch &m1, const DMatch &m2)
                                  { return m1.distance < m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    cout << "-- Max dist:" << max_dist << endl;
    cout << "-- Min dist:" << min_dist << endl;
    
    vector<DMatch> good_matches;
    for (int i = 0; i < descriptor1.rows; i++)
    {
        if (matches[i].distance <= max(2 * min_dist, 30.0)){
            good_matches.push_back(matches[i]);
        }
    }
    
    //-- 第五步:绘制匹配结果
    Mat img_match;
    Mat img_goodmatch;
    drawMatches(img1, keypoint1, img2, keypoint2, matches, img_match);
    drawMatches(img1, keypoint1, img2, keypoint2, good_matches, img_goodmatch);
    imshow("all matches", img_match);
    imshow("good matches", img_goodmatch);
    waitKey(0);

    return 0;
}