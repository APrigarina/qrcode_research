#include <opencv2/opencv.hpp>
#include <iostream>


int main(int argc, char** argv )
{
    std::string image_path = std::string(argv[1]);
    cv::Mat src = cv::imread(image_path);
    cv::QRCodeDetector qrcode;
    cv::Mat corners;
#ifdef HAVE_QUIRC
    std::vector<cv::String> decoded_info;
    std::vector<cv::Mat> straight_barcode;
    qrcode.detectAndDecodeMulti(src, decoded_info, corners, straight_barcode);
#else
    qrcode.detectMulti(src, corners);
#endif

    std::cout << "corners " << corners.size() << std::endl;
    std::cout << "qr codes " << corners.rows << std::endl; 

    cv::Mat inputImage = cv::Mat(src);
    for(int i = 0; i < corners.rows; i++)
    {
        cv::line(inputImage, corners.at<cv::Point2f>(i, 0), corners.at<cv::Point2f>(i, 1), cv::Scalar(100,250,0),5);
        cv::line(inputImage, corners.at<cv::Point2f>(i, 1), corners.at<cv::Point2f>(i, 2), cv::Scalar(100,250,0),5);
        cv::line(inputImage, corners.at<cv::Point2f>(i, 2), corners.at<cv::Point2f>(i, 3), cv::Scalar(100,250,0),5);
        cv::line(inputImage, corners.at<cv::Point2f>(i, 3), corners.at<cv::Point2f>(i, 0), cv::Scalar(100,250,0),5);
    }
    cv::imwrite("test_result.png", inputImage);

    return 0;   
}