
#include <cstddef>
#include <string>
#include <iostream>
#include <iterator>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <limits>
#include <cmath>
#include <queue>

#ifdef HAVE_QUIRC
#include "opencv2/quirc.h"
#endif

#define PI 3.14159265

using namespace std;
using namespace cv;


Mat hull_mask;
Mat bin_barcode;
Point2f intersectionLines(Point2f a1, Point2f a2, Point2f b1, Point2f b2)
{
    Point2f result_square_angle(
                              ((a1.x * a2.y  -  a1.y * a2.x) * (b1.x - b2.x) -
                               (b1.x * b2.y  -  b1.y * b2.x) * (a1.x - a2.x)) /
                              ((a1.x - a2.x) * (b1.y - b2.y) -
                               (a1.y - a2.y) * (b1.x - b2.x)),
                              ((a1.x * a2.y  -  a1.y * a2.x) * (b1.y - b2.y) -
                               (b1.x * b2.y  -  b1.y * b2.x) * (a1.y - a2.y)) /
                              ((a1.x - a2.x) * (b1.y - b2.y) -
                               (a1.y - a2.y) * (b1.x - b2.x))
                              );
    return result_square_angle;
}

vector<Point2f> separateVerticalLines(Mat &img, Mat &bin_barcode, const vector<Vec3d> &list_lines)
{
    vector<Vec3d> result;
    vector<Point2f> point2f_result;

    uint8_t next_pixel;
    vector<double> test_lines;
    double eps_vertical = 0.2;
    double eps_horizontal = 0.1;

    for (int coeff_epsilon = 1; coeff_epsilon < 4; coeff_epsilon++)
    {
        result.clear();
        point2f_result.clear();

        int temp_length = 0;

        cout<<"coeff = "<<coeff_epsilon<<endl;

        for (size_t pnt = 0; pnt < list_lines.size(); pnt++)
        {
            const int x = cvRound(list_lines[pnt][0] + list_lines[pnt][2] * 0.5);
            const int y = cvRound(list_lines[pnt][1]);

            // --------------- Search vertical up-lines --------------- //

            test_lines.clear();
            uint8_t future_pixel_up = 255;

            for (int j = y; j < bin_barcode.rows - 1; j++)
            {
                next_pixel = bin_barcode.ptr<uint8_t>(j + 1)[x];
                temp_length++;
                if (next_pixel == future_pixel_up)
                {
                    future_pixel_up = 255 - future_pixel_up;
                    test_lines.push_back(temp_length);
                    temp_length = 0;
                    if (test_lines.size() == 3) { break; }
                }
            }

            // --------------- Search vertical down-lines --------------- //

            uint8_t future_pixel_down = 255;
            for (int j = y; j >= 1; j--)
            {
                next_pixel = bin_barcode.ptr<uint8_t>(j - 1)[x];
                temp_length++;
                if (next_pixel == future_pixel_down)
                {
                    future_pixel_down = 255 - future_pixel_down;
                    test_lines.push_back(temp_length);
                    temp_length = 0;
                    if (test_lines.size() == 6) { break; }
                }
            }

            // --------------- Compute vertical lines --------------- //

            if (test_lines.size() == 6)
            {
                double length = 0.0, weight = 0.0;

                for (size_t i = 0; i < test_lines.size(); i++) { length += test_lines[i]; }

                CV_Assert(length > 0);
                for (size_t i = 0; i < test_lines.size(); i++)
                {
                    if (i % 3 != 0) { weight += fabs((test_lines[i] / length) - 1.0/ 7.0); }
                    else            { weight += fabs((test_lines[i] / length) - 3.0/14.0); }
                }


                if (weight < eps_horizontal * coeff_epsilon)
                {
                  result.push_back(list_lines[pnt]);
                }

            }
        }
        // cout<<"result size = "<<result.size()<<endl;
        // std::cout << "meow" << '\n';
        if (result.size() > 5)
        {

            for (size_t i = 0; i < result.size(); i++)
            {
                point2f_result.push_back(
                      Point2f(static_cast<float>(result[i][0] + result[i][2] * 0.5),
                              static_cast<float>(result[i][1])));
            }

            vector<Point2f> centers;
            Mat labels;
            double compactness, prev_compactness;
            // if (coeff_epsilon == 0.1)
            // {
            //   prev_compactness = kmeans(point2f_result, 3, labels,
            //        TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1),
            //        3, KMEANS_PP_CENTERS, centers);
            // }
            // else {prev_compactness = compactness;}

            compactness = kmeans(point2f_result, 6, labels,
                 TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1),
                 6, KMEANS_PP_CENTERS, centers);

            // if (compactness == 0) { continue; }

            //
            // std::cout << "eps = " << eps_horizontal * coeff_epsilon << '\n';
            // std::cout << "prev_comp = " << prev_compactness << '\n';
            // std::cout << "comp = " << compactness << '\n';
            // std::cout << "(compactness / prev_compactness ) = " << (compactness / prev_compactness ) << '\n';

            if (compactness > 0)  { break; }
            // if ( (compactness / prev_compactness ) > 1) break;
            // if (compactness == 183119) {break;}

        }
    //     if (flag) break;
    //

    // cout<<result.size()<<endl;
    //


    }
    // for(size_t i = 0; i < result.size()-1; i++)
    // {
    //   line(img, Point2i(result.at(i)[0] + result[i][2] * 0.5, result.at(i)[1]), Point2i(result.at(i)[0] + result[i][2] * 0.5, result.at(i)[1] + result.at(i)[3]
    // ), Scalar(255,0,0), 1);
    // }
    // namedWindow( "lines", WINDOW_NORMAL );
    // imshow( "lines", img );
    // waitKey(0);
    // vector<Point2f> point2f_result;
    // for (size_t i = 0; i < result.size(); i++)
    // {
    //     point2f_result.push_back(
    //           Point2f(static_cast<float>(result[i][0] + result[i][2] * 0.5),
    //                   static_cast<float>(result[i][1])));
    // }
    //
    for (int j=0; j < point2f_result.size(); j++)
    {
      circle(img, point2f_result.at(j), 3, Scalar(63,63,214), -1);
    }
    imwrite( "result_images/intersection_points.png", img );
    // namedWindow( "intersection_points", WINDOW_NORMAL );
    // imshow( "intersection_points", img );
    // waitKey(0);
    //
    // std::cout<<"intersection_points\n"<<point2f_result<<std::endl;
    //
    // vector<Point2f> centers;
    // Mat labels;
    // double compactness, next_compactness;
    //
    //
    // // for (int i = 1; i < point2f_result.size(); i++)
    // // {
    // //   compactness = kmeans(point2f_result, i, labels,
    // //          TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1),
    // //          i, KMEANS_PP_CENTERS, centers);
    // //   // cout<<centers<<endl;
    // //   cout<<i<<" compactness = "<<compactness<<endl;
    // // }
    //
    // size_t num_qr, num_points;
    // for(num_points = 1; num_points < point2f_result.size(); num_points++)
    // {
    //
    //     if(num_points == 1)
    //     {
    //        compactness = kmeans(point2f_result, num_points, labels,
    //               TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1),
    //               num_points, KMEANS_PP_CENTERS, centers);
    //
    //        num_points++;
    //     }
    //     else compactness = next_compactness;
    //
    //     next_compactness = kmeans(point2f_result, num_points, labels,
    //           TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1),
    //           num_points, KMEANS_PP_CENTERS, centers);
    //     // compactness_frac.push_back(compactness / next_compactness);
    //     cout<<num_points<<" compactness = "<<compactness / next_compactness<<endl;
    //     cout<<centers.size()<<endl;
    // }

    // if (centers.size() != 3) { cout<<"meow"<<endl;}

    return point2f_result;
}
void fixationPoints(Mat &bin_barcode, vector<Point2f> &local_point)
{
    double eps_vertical = 0.2;
    double eps_horizontal = 0.1;

    double cos_angles[3], norm_triangl[3];


    norm_triangl[0] = norm(local_point[1] - local_point[2]);
    norm_triangl[1] = norm(local_point[0] - local_point[2]);
    norm_triangl[2] = norm(local_point[1] - local_point[0]);


    cos_angles[0] = (norm_triangl[1] * norm_triangl[1] + norm_triangl[2] * norm_triangl[2]
                  -  norm_triangl[0] * norm_triangl[0]) / (2 * norm_triangl[1] * norm_triangl[2]);
    cos_angles[1] = (norm_triangl[0] * norm_triangl[0] + norm_triangl[2] * norm_triangl[2]
                  -  norm_triangl[1] * norm_triangl[1]) / (2 * norm_triangl[0] * norm_triangl[2]);
    cos_angles[2] = (norm_triangl[0] * norm_triangl[0] + norm_triangl[1] * norm_triangl[1]
                  -  norm_triangl[2] * norm_triangl[2]) / (2 * norm_triangl[0] * norm_triangl[1]);

    const double angle_barrier = 0.85;
    if (fabs(cos_angles[0]) > angle_barrier || fabs(cos_angles[1]) > angle_barrier || fabs(cos_angles[2]) > angle_barrier)
    {
        local_point.clear();
        return;
    }

    size_t i_min_cos =
       (cos_angles[0] < cos_angles[1] && cos_angles[0] < cos_angles[2]) ? 0 :
       (cos_angles[1] < cos_angles[0] && cos_angles[1] < cos_angles[2]) ? 1 : 2;

    size_t index_max = 0;
    double max_area = std::numeric_limits<double>::min();
    for (size_t i = 0; i < local_point.size(); i++)
    {
        const size_t current_index = i % 3;
        const size_t left_index  = (i + 1) % 3;
        const size_t right_index = (i + 2) % 3;

        const Point2f current_point(local_point[current_index]),
            left_point(local_point[left_index]), right_point(local_point[right_index]),
            central_point(intersectionLines(current_point,
                              Point2f(static_cast<float>((local_point[left_index].x + local_point[right_index].x) * 0.5),
                                      static_cast<float>((local_point[left_index].y + local_point[right_index].y) * 0.5)),
                              Point2f(0, static_cast<float>(bin_barcode.rows - 1)),
                              Point2f(static_cast<float>(bin_barcode.cols - 1),
                                      static_cast<float>(bin_barcode.rows - 1))));


        vector<Point2f> list_area_pnt;
        list_area_pnt.push_back(current_point);

        vector<LineIterator> list_line_iter;
        list_line_iter.push_back(LineIterator(bin_barcode, current_point, left_point));
        list_line_iter.push_back(LineIterator(bin_barcode, current_point, central_point));
        list_line_iter.push_back(LineIterator(bin_barcode, current_point, right_point));

        for (size_t k = 0; k < list_line_iter.size(); k++)
        {
            uint8_t future_pixel = 255, count_index = 0;
            for(int j = 0; j < list_line_iter[k].count; j++, ++list_line_iter[k])
            {
                if (list_line_iter[k].pos().x >= bin_barcode.cols ||
                    list_line_iter[k].pos().y >= bin_barcode.rows) { break; }
                const uint8_t value = bin_barcode.at<uint8_t>(list_line_iter[k].pos());
                if (value == future_pixel)
                {
                    future_pixel = 255 - future_pixel;
                    count_index++;
                    if (count_index == 3)
                    {
                        list_area_pnt.push_back(list_line_iter[k].pos());
                        break;
                    }
                }
            }
        }

        const double temp_check_area = contourArea(list_area_pnt);
        if (temp_check_area > max_area)
        {
            index_max = current_index;
            max_area = temp_check_area;
        }

    }
    if (index_max == i_min_cos) { std::swap(local_point[0], local_point[index_max]); }
    else { local_point.clear(); return; }


    const Point2f rpt = local_point[0], bpt = local_point[1], gpt = local_point[2];
    Matx22f m(rpt.x - bpt.x, rpt.y - bpt.y, gpt.x - rpt.x, gpt.y - rpt.y);
    if( determinant(m) > 0 )
    {
        std::swap(local_point[1], local_point[2]);
    }
    // std::cout<<"fixation_points\n"<<local_point<<std::endl;
}

bool testBypassRoute(vector<Point2f> hull, int start, int finish)
{
  int index_hull = start, next_index_hull, hull_size = (int)hull.size();
  double test_length[2] = { 0.0, 0.0 };
  do
  {
      next_index_hull = index_hull + 1;
      if (next_index_hull == hull_size) { next_index_hull = 0; }
      test_length[0] += norm(hull[index_hull] - hull[next_index_hull]);
      index_hull = next_index_hull;
  }
  while(index_hull != finish);

  index_hull = start;
  do
  {
      next_index_hull = index_hull - 1;
      if (next_index_hull == -1) { next_index_hull = hull_size - 1; }
      test_length[1] += norm(hull[index_hull] - hull[next_index_hull]);
      index_hull = next_index_hull;
  }
  while(index_hull != finish);

  if (test_length[0] < test_length[1]) { return true; } else { return false; }
}
inline double getCosVectors(Point2f a, Point2f b, Point2f c)
{
    return ((a - b).x * (c - b).x + (a - b).y * (c - b).y) / (norm(a - b) * norm(c - b));
}
// b     
//  \   
//   \ 
//    a 

inline double getAngleClockwise(Point2f a, Point2f b)
{
    double dot, det, result;

    // float vec_x = a.x - b.x;
    // float vec_y = a.y - b.y;
    // // float vec_y = b.y - a.y;

    // float base_x = bin_barcode.cols - b.x;
    // float base_y = bin_barcode.rows - b.y;
    // // float base_y = b.y - bin_barcode.rows;

    // dot = base_x * vec_x + base_y * vec_y;
    // det = base_x * vec_y - base_y * vec_x;
    // result = atan2(det, dot) * 180.0 / PI;
    // if (result < 0)
    // {
    //     result = 180.0 + result;
    // }

    result = atan2(a.y, a.x) * 180.0 / PI;

    return result;
}
//     a
//      \
//    b__\______________c
//
//
//
float distancePt(Point2f a, Point2f b , Point2f c)
{
    float A, B, C, result;
    A = c.y - b.y;
    B = c.x - b.x;
    C = c.x * b.y - b.x * c.y;
    result = (A * a.x - B * a.y + C) / sqrt(A*A + B*B);

    return result;
}

double pointPosition(Point2f a, Point2f b , Point2f c)
{
    double result;
    result = (a.x - b.x) * (c.y - b.y) - (c.x - b.x) * (a.y - b.y);

    return result;
}

Point2f getMidSection(Point2f a, Point2f b)
{
    float x = (a.x + b.x) / 2;
    float y = (a.y + b.y) / 2;
    return Point2f(x, y);
}

Point2f pointProjection(Point2f a1, Point2f a2, Point2f projectedPoint)
{

  float x=((a2.x-a1.x)*(a2.y-a1.y)*(projectedPoint.y-a1.y)+a1.x*pow(a2.y-a1.y, 2)+projectedPoint.x*pow(a2.x-a1.x, 2))/(pow(a2.y-a1.y, 2)+pow(a2.x-a1.x, 2));
  float y=(a2.y-a1.y)*(x-a1.x)/(a2.x-a1.x)+a1.y;

  Point2f projection = Point2f(x, y);

  return projection;
}
double lagrange(int y, int n, vector<int> &x_arr, vector<int> &y_arr)
{
    bool print = y == 620;
    // cout<<"y = "<<y<<endl;
    double sum = 0;
    for (int i = 0; i < n; i++)
    {
        // cout<<" y_arr[i]"<<y_arr[i]<<endl;
        double l = 1.0;
        for (int j = 0; j < n; j++)
        {
            if (i != j)
            {
                // cout<<" y_arr[j]"<<y_arr[j]<<endl;
                // l *= (y - y_arr.at(j)) / (y_arr.at(i) - y_arr.at(j));
                double a = (y - y_arr.at(j));
                double b = (y_arr.at(i) - y_arr.at(j));
                double c = a / b;
                if (print && i == 0)
                {
                    cout<<"before l "<<l<<endl;
                    printf("y %d\ny_arr[i] %d\ny_arr[j] %d\n", y, y_arr.at(i), y_arr.at(j));
                    printf("c=%lf, l=%lf, l*c=%lf\n", c, l, l*c); 
                }
                l = l * c;
                if (print && i == 0)
                    cout<<"after l "<<l<<endl;
                // int count = 0;
                // if (l <= 0)
                // {
                //     count++;
                //     if (count > 1) continue;
                //     cout<<"y "<<y<<endl;
                //     cout<<"y_arr[i] "<<y_arr.at(i)<<endl;
                //     cout<<"y_arr[j] "<<y_arr.at(j)<<endl;
    
                //     cout<<"after l "<<l<<endl;
                //     cout<<c<<endl;
                // }
                
            }
           
        }
        sum += x_arr[i] * l;
        // if (l <= 0 && print && i == 0)
        // {
        if(print) 
        {
            cout<<"i:"<<i<<" l "<<l<<" sum "<<sum<<endl;
        }
        // }
        
    }
    return sum;
}

vector<vector<double>> spline(vector<int> &x_arr, vector<int> &y_arr)
{
    int n = y_arr.size(); 
    vector<int> a;
    vector<double> b(n-1), d(n-1), h(n-1), alpha(n-1), c(n), l(n), mu(n), z(n);

    for (int i = 0; i < x_arr.size(); i++)
    {
        a.push_back(x_arr.at(i));
    }
    // cout<<"a\n";
    for (int i = 0; i < n-1; i++)
    {
        h.at(i) = y_arr.at(i+1) - y_arr.at(i);
    }
    // cout<<"h\n";
    for (int i = 1; i < n-1; i++)
    {
        alpha.at(i) = 3 / h.at(i) * (a.at(i+1) - a.at(i)) - 3 / (h.at(i-1)) * (a.at(i) - a.at(i-1));
    }
    // cout<<"alpha\n";
    l.at(0) = 1;
    mu.at(0) = 0;
    z.at(0) = 0;
  
    for (int i = 1; i < n-1; i++)
    {
        l.at(i) = 2 * (y_arr.at(i+1) - y_arr.at(i-1)) - h.at(i-1) * mu.at(i-1);
        mu.at(i) = h.at(i) / l.at(i);
        z.at(i) = (alpha.at(i) - h.at(i-1) * z.at(i-1)) / l.at(i);
    }
    // cout<<"l mu z\n";
    l.at(n-1) = 1;
    z.at(n-1) = 0;
    c.at(n-1) = 0;

    for(int j = n-2; j >= 0; j--)
    {
        c.at(j) = z.at(j) - mu.at(j) * c.at(j+1);
        b.at(j) = (a.at(j+1) - a.at(j)) / h.at(j) - (h.at(j) * (c.at(j+1) + 2 * c.at(j))) / 3;
        d.at(j) = (c.at(j+1) - c.at(j)) / (3 * h.at(j));
    }
    // cout<<"c b d\n";

    vector<vector<double>> S(n-1);
    for (int i = 0; i < n-1; i++)
    {
        S.at(i).push_back(a.at(i));
        S.at(i).push_back(b.at(i));
        S.at(i).push_back(c.at(i));
        S.at(i).push_back(d.at(i));
    }
    // cout<<"S\n";

    return S;

}

bool sortPairAsc(const pair<size_t, double> &a, 
               const pair<size_t, double> &b) 
{ 
    return (a.second > b.second); 
} 
bool sortPairDesc(const pair<size_t, double> &a, 
                  const pair<size_t, double> &b) 
{ 
    return (a.second < b.second); 
}
bool sortPointsByX(const Point a, Point b)
{
    return (a.x < b.x);
} 
bool sortPointsByY(const Point a, Point b)
{
    return (a.y < b.y);
}
vector<Point> get_points(Mat&img, vector<Point>&side, int start, int end, int step)
{
    vector<Point> points;
    Point a1, a2, a3;

    double max_neighbour_angle = 1.0;
    cout<<"size "<<side.size()<<endl;
    
    if(side.size() < 3)
    {
        for(int i = 0; i < side.size(); i++)
        {
            points.push_back(side.at(i));
        }
        return points;
    }
    
    printf("start = %d, end = %d, step = %d\n", start, end, step);
    for (int j = start; j != end; j+=step)
    {
        printf("j = %d, j+step = %d, j+step*2 = %d\n", j, j+step, j+step*2);

        a1 = side.at(j);
        a2 = side.at(j+step);
        a3 = side.at(j+step*2);
        

        double neighbour_angle = getCosVectors(a1, a2, a3);
        neighbour_angle = floor( neighbour_angle*1000 )/1000;
        // max_neighbour_angle = acos(floor( max_neighbour_angle*100 )/100)*180.0/PI;
        cout<<"neighbour_angle "<<neighbour_angle<<endl;
        cout<<"max_neighbour_angle "<<max_neighbour_angle<<endl;
        // circle(img, a2, 1, Scalar(0, 255, 255), -1);
        // putText(img, to_string(neighbour_angle), a2, 1, 1, Scalar(0,255,255), 1);
        cout<<"a1 "<<a1<<" a2 "<<a2<<endl;
        // if(side.size()==5&&(j==4))
        // {
        //     circle(img, a2, 1, Scalar(0,255,255), -1);
        //     putText(img, "a2", a2, 1, 1, Scalar(0,255,255), 1);
        //     // putText(img, "b2", a2, 1, 1, Scalar(0,255,255), 1);
        //     // putText(img, "b3", a3, 1, 1, Scalar(0,255,255), 1);
        // }

        if (neighbour_angle < max_neighbour_angle)
        {
            max_neighbour_angle = neighbour_angle;
            // if (j+step == end)
            // {
            //     Point temp_a1 = side.at(j+step);
            //     Point temp_a2 = side.at(j+step*2); 
            //     if(norm(temp_a1-temp_a2) > 5)
            //     {
            //         a1 = temp_a1;
            //         a2 = temp_a2;
            //     }
                 
            // }

        }
        else  
        {
            a1 = side.at(j-step);
            a2 = side.at(j); 
            cout<<"a2 a1 "<<norm(a2-a1)<<endl;
            if (norm(a2-a1) > 5 )
            {
                bool flag = false;
                for (size_t i = j; i != end-step; i+=step)
                {
                    a3 = side.at(j+step);
                    cout<<"a2 a3 "<<norm(a2-a3)<<endl;   
                    if (norm(a2-a3) > 5)
                    {
                        flag = true;
                        break;
                    }                            
                }
                if(flag) 
                {
                    cout<<"krya\n";
                    break;
                }
            }
        }

    }


    points.push_back(a1);
    points.push_back(a2);
    if (a2 != a3)
        points.push_back(a3);
    return points;
}

// vector<pair<int,vector<Point>>> get_marker_points(vector<pair<int,vector<Point>>> cur_sides)
// {
//     vector<vector<Point>> marker_points;
//     bool make_full_side_first = false;
//     for (int i = 0; i < cur_indexes.size(); i++)
//     {
//         int idx = cur_indexes[i];
//         vector<size_t> indexes;
//         for (size_t j = 0; j < markers_edge_points.size(); j++)
//         {
//             for (size_t k = 0; k < markers_edge_points[j].size(); k++)
//             {
//                 if (norm(Point(cvRound(result_angle_list[idx].x), cvRound(result_angle_list[idx].y)) - markers_edge_points[j][k]) < 5) 
//                 {
//                     indexes.push_back(j);
//                 }
//                 if (norm(Point(cvRound(result_angle_list[(idx+1)%4].x), cvRound(result_angle_list[(idx+1)%4].y)) - markers_edge_points[j][k]) < 5) 
//                 {
//                     indexes.push_back(j); 
//                 }
//             }
//         }
//         cout<<"indexes size "<<indexes.size()<<endl;
//         if (i == 0)
//         {
//             if (indexes.size() == 1)
//             {
//                 make_full_side_first = true;
//             }
//         }
//         for (int h = 0; h < indexes.size(); h++)
//         {
//             vector<pair<size_t, double>> id_dist;
//             vector<Point> points;
//             for (size_t k = 0; k < markers_edge_points[indexes[h]].size(); k++)
//             {
//                 double dist;
//                 dist = norm(markers_edge_points[indexes[h]][k] - Point(cvRound(result_angle_list[idx].x), cvRound(result_angle_list[idx].y)));
//                 dist += norm(markers_edge_points[indexes[h]][k] - Point(cvRound(result_angle_list[(idx+1)%4].x), cvRound(result_angle_list[(idx+1)%4].y)));
//                 dist /= 2;
//                 id_dist.push_back(pair<size_t, double>(k, dist));

//             }
//             if(!id_dist.empty())

//             {

//                 sort(id_dist.begin(), id_dist.end(), sortPairDesc);
//                 cout<<"marker "<<h<<endl;
//                 for (int s = 0; s < id_dist.size(); s++)
//                 {
//                     cout << id_dist[s].first << ": " << id_dist[s].second << endl;
//                     putText(image, to_string(id_dist[s].first), markers_edge_points[indexes[h]][id_dist[s].first], 1, 1, Scalar(0,0,255), 1);
//                 }
//                 Point p1, p2;
//                 int index_p1, index_p2;
//                 for (int r = 4; r > 0; r--)
//                 {
//                     if((id_dist[0].first == r%4) && (id_dist[1].first == (r-1)%4))
//                     {
//                         index_p1 = id_dist[0].first;
//                         index_p2 = id_dist[1].first;
                        
//                     }
//                     else if((id_dist[1].first == r%4) && (id_dist[0].first == (r-1)%4))
//                     {
//                         index_p1 = id_dist[1].first;
//                         index_p2 = id_dist[0].first;
                        
//                     }
//                 }
                
//                 p1 = markers_edge_points[indexes[h]][index_p1];
//                 p2 = markers_edge_points[indexes[h]][index_p2];
//                 putText(image, "p1", p1, 1, 2, Scalar(255), 1);
//                 putText(image, "p2", p2, 1, 2, Scalar(255), 1);

//                 int index1, index2;
//                 for(int n = 0; n < hull_points[indexes[h]].size(); n++)
//                 {
                    
//                     if (hull_points[indexes[h]][n] == p1)
//                     {
//                         index1 = n;
//                     }
//                     if (hull_points[indexes[h]][n] == p2)
//                     {
//                         index2 = n;
//                     }
//                 }

//                 if (index1 > index2)
//                 {

//                     for (int l = index1; l < hull_points[indexes[h]].size(); l++)
//                     {
//                         // circle(image, hull_points[indexes[h]][l], 3, Scalar(0,255,0), -1);
//                         // putText(image, to_string(l), hull_points[indexes[h]][l], 1, 1, Scalar(29, 50, 138), 1);
//                         points.push_back(hull_points[indexes[h]][l]);
//                     }
//                     for (int l = 0; l <= index2; l++)
//                     {
//                         // circle(image, hull_points[indexes[h]][l], 3, Scalar(0,255,0), -1);
//                         // putText(image, to_string(l), hull_points[indexes[h]][l], 1, 1, Scalar(29, 50, 138), 1);

//                         points.push_back(hull_points[indexes[h]][l]);
                    
//                     }

//                 }
//                 else
//                 {
//                     for (int l = index1; l <= index2; l++)
//                     {
//                     //     circle(image, hull_points[indexes[h]][l], 3, Scalar(0,255,0), -1);
//                     //     putText(image, to_string(l), hull_points[indexes[h]][l], 1, 1, Scalar(29, 50, 138), 1);

//                         points.push_back(hull_points[indexes[h]][l]);
                    
//                     }
//                 }
//                 // points.push_back(p2);
//                 if (abs(p1.x - p2.x) > abs(p1.y - p2.y))
//                 {
//                     sort(points.begin(), points.end(), sortPointsByX);
//                 }
//                 else
//                 {
//                     sort(points.begin(), points.end(), sortPointsByY);
//                 }

//             }
//             cout<<"points size "<<points.size()<<endl;
//             marker_points.push_back(points);

//             printf("idx %d points size %d\n", idx, points.size());
//             for (int p = 0; p < points.size(); p++)
//                 circle(add_points_image, points[p], 2, Scalar(100, 250, 0), -1);

//         }

// }

vector<Point2f> getQuadrilateral(Mat &image, Mat &bin_barcode, Mat &original_img, vector<vector<Point>> &hull_points, vector<Point2f> &angle_list, char** argv)
{

    Mat spline_image = image.clone();
    Mat arc_image = image.clone();
    Mat add_points_image = image.clone();
    Mat lagrange_image = image.clone();
    Mat convexhull_image = image.clone();
    Mat transform_image = image.clone();
    Mat lines_image = image.clone();
    Mat side_points = image.clone();
    Mat close_points = image.clone();

// построение эллипса
    size_t angle_size = angle_list.size();

    Mat angle_mask = Mat::zeros(bin_barcode.rows, bin_barcode.cols, CV_8UC1);
    for (size_t i = 0; i < angle_size; i++)
    {
        line(angle_mask, angle_list[i%angle_size], angle_list[(i+1)%angle_size], Scalar(255), 2);
    } 
    // namedWindow("angle_mask", WINDOW_NORMAL);
    // imshow("angle_mask", angle_mask);
    // waitKey(0);
    vector<vector<Point>> angle_contours;
    findContours( angle_mask, angle_contours,  RETR_EXTERNAL , CHAIN_APPROX_NONE, Point(0, 0) );
    

    vector<Point> locations;
    Mat ellipse_mask = Mat::zeros(bin_barcode.rows, bin_barcode.cols, CV_8UC1);

    Point2f centerPoint = intersectionLines(angle_list[0], angle_list[2],
                                        angle_list[1], angle_list[3]);



    int center_x = cvRound(centerPoint.x);
    int center_y = cvRound(centerPoint.y);
    // for (size_t i = 0; i < angle_size; i++)
    // {
    //     line(ellipse_mask, angle_list[i%angle_size], angle_list[(i+1)%angle_size], Scalar(255), 2);
    // } 

    // for (int i = 0; i < angle_size; i++)
    // {
    //     Point2f starting_axes_point = getMidSection(angle_list[i], angle_list[(i+1)%angle_size]);
    //     double first_axis = norm(centerPoint - starting_axes_point);

    //     double starting_angle = getAngleClockwise(starting_axes_point, centerPoint);
    // starting_angle = 80.0;
    //     Point2f ending_axes_point = getMidSection(angle_list[(i+1)%angle_size], angle_list[(i+2)%angle_size]);
    //     double second_axis = norm(centerPoint - ending_axes_point);
    //     double ending_angle = getAngleClockwise(ending_axes_point, centerPoint);
    // ending_angle = 180.0;
    //     // double rotation_angle = acos(getCosVectors(starting_axes_point, centerPoint, ending_axes_point));
    //     double rotation_angle = ending_angle - starting_angle;
    //     std::cout<<"starting_axes_point "<<starting_axes_point<<" ending_axes_point "<<ending_axes_point<<std::endl;
    //     circle(ellipse_mask, starting_axes_point, 10, Scalar(255), -1);
    //     circle(ellipse_mask, ending_axes_point, 10, Scalar(255), -1);
    //     printf("first_axis %f, second_axis %f\n", first_axis, second_axis);
    //     printf("starting_angle %f, ending_angle %f\n", starting_angle, ending_angle);
    //     printf("rotation_angle %f\n", rotation_angle);

    //     ellipse(ellipse_mask, centerPoint, Size(first_axis,second_axis), rotation_angle, starting_angle, ending_angle, Scalar(255), -1);
    //     String name = "result_images/ellipse_mask_" + to_string(i) + ".png";
    //     imwrite(name, ellipse_mask);

    // }

    vector<Point> axes_points;
    for (int y = center_y; y > 0; y--)
    {
        uint8_t mask_pixel = angle_mask.ptr<uint8_t>(y)[center_x];
        if (mask_pixel == 255)
        {
            axes_points.push_back(Point(center_x, y));
            break;
        }
    }

    for (int x = center_x; x > 0; x--)
    {
        uint8_t mask_pixel = angle_mask.ptr<uint8_t>(center_y)[x];
        if (mask_pixel == 255)
        {
            axes_points.push_back(Point(x, center_y));
            break;
        }
    }
    for (int y = center_y; y < angle_mask.rows; y++)
    {
        uint8_t mask_pixel = angle_mask.ptr<uint8_t>(y)[center_x];
        if (mask_pixel == 255)
        {
            axes_points.push_back(Point(center_x, y));
            break;
        }
    }
    for (int x = center_x; x < angle_mask.cols; x++)
    {
        uint8_t mask_pixel = angle_mask.ptr<uint8_t>(center_y)[x];
        if (mask_pixel == 255)
        {
            axes_points.push_back(Point(x, center_y));
            break;
        }
    }

    for (int i = 0; i < axes_points.size(); i++)
    {
        if (i%2==0)
        {
            int temp_x = abs(cvRound(centerPoint.x - axes_points[(i+1)%axes_points.size()].x));
            int temp_y = abs(cvRound(centerPoint.y - axes_points[i%axes_points.size()].y));
            ellipse(ellipse_mask, centerPoint, Size(temp_y,temp_x), 90, 90+i*90, 180+i*90, Scalar(255), -1);
        }
        else
        {
            int temp_y = abs(cvRound(centerPoint.y - axes_points[(i+1)%axes_points.size()].y));
            int temp_x = abs(cvRound(centerPoint.x - axes_points[i%axes_points.size()].x));
            ellipse(ellipse_mask, centerPoint, Size(temp_y,temp_x), -90, 90+i*90, 180+i*90, Scalar(255), -1);        
        }
        
    }

    



    imwrite("result_images/ellipse_mask.png", ellipse_mask);

    // namedWindow("ellipse_mask", WINDOW_NORMAL);
    // imshow("ellipse_mask", ellipse_mask);
    // waitKey(0);

    // Mat sum = angle_mask + ellipse_mask;
    // namedWindow("sum", WINDOW_NORMAL);
    // imshow("sum", sum);
    // imwrite("sum.png", sum);
    // waitKey(0);

    vector<Point2f> ellipse_points;
    vector<Point2f> true_ellipse_points;
    findNonZero(ellipse_mask, ellipse_points);

    Mat true_ellipse_mask = Mat::zeros(bin_barcode.rows, bin_barcode.cols, CV_8UC1);

    for (int i = 0; i < angle_contours.size(); i++)
    {
        for (int j = 0; j < ellipse_points.size(); j++)
        {
            if (pointPolygonTest(angle_contours[i], ellipse_points[j], false) >= 0)
            {
                true_ellipse_points.push_back(ellipse_points[j]);
                circle(true_ellipse_mask, ellipse_points[j], 1, Scalar(255), -1);
            }
        }
    }
    imwrite("result_images/true_ellipse_mask.png", true_ellipse_mask);

    ellipse_points = true_ellipse_points;



    Mat fill_bin_barcode_0 = bin_barcode.clone();
    Mat mask_ellipse = Mat::zeros(bin_barcode.rows + 2, bin_barcode.cols + 2, CV_8UC1);
printf("mask_ellipse %lu * %lu\n", mask_ellipse.rows, mask_ellipse.cols);
printf("bin_barcode %lu * %lu\n", bin_barcode.rows, bin_barcode.cols);

    for(int i = 0; i < ellipse_points.size(); i++)
    {
        uint8_t value = bin_barcode.at<uint8_t>(ellipse_points.at(i));
        uint8_t mask_value = mask_ellipse.at<uint8_t>(ellipse_points.at(i) + Point2f(1.0, 1.0));
        if (value == 0 && mask_value ==0)
        {
            floodFill(fill_bin_barcode_0, mask_ellipse, ellipse_points.at(i), 255,
                        0, Scalar(), Scalar(), FLOODFILL_MASK_ONLY);
        }
    }
Mat mask_roi_ellipse = mask_ellipse(Range(1, mask_ellipse.rows - 1), Range(1, mask_ellipse.cols - 1));
printf("bin_barcode %lu * %lu\n", bin_barcode.rows, bin_barcode.cols);
printf("mask_roi_ellipse %lu * %lu\n", mask_roi_ellipse.rows, mask_roi_ellipse.cols);
printf("hull_mask %lu * %lu\n", hull_mask.rows, hull_mask.cols);
    // Mat temp = hull_mask;
std::cout<<"krya\n";

    findNonZero(mask_roi_ellipse, locations);
    for (int j=0; j < locations.size(); j++)
    {
        circle(mask_roi_ellipse, locations.at(j), 0.2, Scalar(255,0,0));
    }

    Mat result_mat = mask_roi_ellipse + hull_mask;

    mask_roi_ellipse = result_mat.clone();



    imwrite( "result_images/integer_hull_mask.png", mask_roi_ellipse );
    // namedWindow( "integer_hull mask", WINDOW_NORMAL );
    // imshow( "integer_hull mask", mask_roi_ellipse );
    // waitKey(0);

    vector<Point> integer_hull;
    convexHull(locations, integer_hull);


    int hull_size = (int)integer_hull.size();
    vector<Point2f> hull(hull_size);
    for (int i = 0; i < hull_size; i++)
    {
        float x = saturate_cast<float>(integer_hull[i].x);
        float y = saturate_cast<float>(integer_hull[i].y);
        hull[i] = Point2f(x, y);
    }

    const double experimental_area = fabs(contourArea(hull));

    vector<Point2f> result_hull_point(angle_size);
    double min_norm;
    for (size_t i = 0; i < angle_size; i++)
    {
        min_norm = std::numeric_limits<double>::max();
        Point closest_pnt;
        for (int j = 0; j < hull_size; j++)
        {
            double temp_norm = norm(hull[j] - angle_list[i]);
            if (min_norm > temp_norm)
            {
                min_norm = temp_norm;
                closest_pnt = hull[j];
            }
        }
        result_hull_point[i] = closest_pnt;
    }


    int start_line[2] = { 0, 0 }, finish_line[2] = { 0, 0 }, unstable_pnt = 0;
    for (int i = 0; i < hull_size; i++)
    {
        if (result_hull_point[2] == hull[i]) { start_line[0] = i; }
        if (result_hull_point[1] == hull[i]) { finish_line[0] = start_line[1] = i; }
        if (result_hull_point[0] == hull[i]) { finish_line[1] = i; }
        if (result_hull_point[3] == hull[i]) { unstable_pnt = i; }
    }

    int index_hull, extra_index_hull, next_index_hull, extra_next_index_hull;
    Point result_side_begin[4], result_side_end[4];

    bool bypass_orientation = testBypassRoute(hull, start_line[0], finish_line[0]);

    min_norm = std::numeric_limits<double>::max();
    index_hull = start_line[0];
    do
    {
        if (bypass_orientation) { next_index_hull = index_hull + 1; }
        else { next_index_hull = index_hull - 1; }

        if (next_index_hull == hull_size) { next_index_hull = 0; }
        if (next_index_hull == -1) { next_index_hull = hull_size - 1; }

        Point angle_closest_pnt =  norm(hull[index_hull] - angle_list[1]) >
        norm(hull[index_hull] - angle_list[2]) ? angle_list[2] : angle_list[1];

        Point intrsc_line_hull =
        intersectionLines(hull[index_hull], hull[next_index_hull],
                            angle_list[1], angle_list[2]);
        double temp_norm = getCosVectors(hull[index_hull], intrsc_line_hull, angle_closest_pnt);
        if (min_norm > temp_norm &&
            norm(hull[index_hull] - hull[next_index_hull]) >
            norm(angle_list[1] - angle_list[2]) * 0.1)
        {
            min_norm = temp_norm;
            result_side_begin[0] = hull[index_hull];
            result_side_end[0]   = hull[next_index_hull];
        }


        index_hull = next_index_hull;
    }
    while(index_hull != finish_line[0]);

    if (min_norm == std::numeric_limits<double>::max())
    {
        result_side_begin[0] = angle_list[1];
        result_side_end[0]   = angle_list[2];
    }

    min_norm = std::numeric_limits<double>::max();
    index_hull = start_line[1];
    bypass_orientation = testBypassRoute(hull, start_line[1], finish_line[1]);
    do
    {
        if (bypass_orientation) { next_index_hull = index_hull + 1; }
        else { next_index_hull = index_hull - 1; }

        if (next_index_hull == hull_size) { next_index_hull = 0; }
        if (next_index_hull == -1) { next_index_hull = hull_size - 1; }

        Point angle_closest_pnt =  norm(hull[index_hull] - angle_list[0]) >
        norm(hull[index_hull] - angle_list[1]) ? angle_list[1] : angle_list[0];

        Point intrsc_line_hull =
        intersectionLines(hull[index_hull], hull[next_index_hull],
                            angle_list[0], angle_list[1]);
        double temp_norm = getCosVectors(hull[index_hull], intrsc_line_hull, angle_closest_pnt);
        if (min_norm > temp_norm &&
            norm(hull[index_hull] - hull[next_index_hull]) >
            norm(angle_list[0] - angle_list[1]) * 0.05)
        {
            min_norm = temp_norm;
            result_side_begin[1] = hull[index_hull];
            result_side_end[1]   = hull[next_index_hull];
        }

        index_hull = next_index_hull;
    }
    while(index_hull != finish_line[1]);

    if (min_norm == std::numeric_limits<double>::max())
    {
        result_side_begin[1] = angle_list[0];
        result_side_end[1]   = angle_list[1];
    }

    bypass_orientation = testBypassRoute(hull, start_line[0], unstable_pnt);
    const bool extra_bypass_orientation = testBypassRoute(hull, finish_line[1], unstable_pnt);

    vector<Point2f> result_angle_list(4), test_result_angle_list(4);
    double min_diff_area = std::numeric_limits<double>::max();
    index_hull = start_line[0];
    const double standart_norm = std::max(
        norm(result_side_begin[0] - result_side_end[0]),
        norm(result_side_begin[1] - result_side_end[1]));
    do
    {
        if (bypass_orientation) { next_index_hull = index_hull + 1; }
        else { next_index_hull = index_hull - 1; }

        if (next_index_hull == hull_size) { next_index_hull = 0; }
        if (next_index_hull == -1) { next_index_hull = hull_size - 1; }

        if (norm(hull[index_hull] - hull[next_index_hull]) < standart_norm * 0.1)
        { index_hull = next_index_hull; continue; }

        extra_index_hull = finish_line[1];
        do
        {
            if (extra_bypass_orientation) { extra_next_index_hull = extra_index_hull + 1; }
            else { extra_next_index_hull = extra_index_hull - 1; }

            if (extra_next_index_hull == hull_size) { extra_next_index_hull = 0; }
            if (extra_next_index_hull == -1) { extra_next_index_hull = hull_size - 1; }

            if (norm(hull[extra_index_hull] - hull[extra_next_index_hull]) < standart_norm * 0.1)
            { extra_index_hull = extra_next_index_hull; continue; }

            test_result_angle_list[0]
            = intersectionLines(result_side_begin[0], result_side_end[0],
                                result_side_begin[1], result_side_end[1]);
            test_result_angle_list[1]
            = intersectionLines(result_side_begin[1], result_side_end[1],
                                hull[extra_index_hull], hull[extra_next_index_hull]);
            test_result_angle_list[2]
            = intersectionLines(hull[extra_index_hull], hull[extra_next_index_hull],
                                hull[index_hull], hull[next_index_hull]);
            test_result_angle_list[3]
            = intersectionLines(hull[index_hull], hull[next_index_hull],
                                result_side_begin[0], result_side_end[0]);

            const double test_diff_area
                = fabs(fabs(contourArea(test_result_angle_list)) - experimental_area);
            if (min_diff_area > test_diff_area)
            {
                min_diff_area = test_diff_area;
                for (size_t i = 0; i < test_result_angle_list.size(); i++)
                {
                    result_angle_list[i] = test_result_angle_list[i];
                }
            }

            extra_index_hull = extra_next_index_hull;
        }
        while(extra_index_hull != unstable_pnt);

        index_hull = next_index_hull;
    }
    while(index_hull != unstable_pnt);

    // check label points
    if (norm(result_angle_list[0] - angle_list[1]) > 2) { result_angle_list[0] = angle_list[1]; }
    if (norm(result_angle_list[1] - angle_list[0]) > 2) { result_angle_list[1] = angle_list[0]; }
    if (norm(result_angle_list[3] - angle_list[2]) > 2) { result_angle_list[3] = angle_list[2]; }

    // check calculation point
    if (norm(result_angle_list[2] - angle_list[3]) >
        (norm(result_angle_list[0] - result_angle_list[1]) +
        norm(result_angle_list[0] - result_angle_list[3])) * 0.5 )
    { result_angle_list[2] = angle_list[3]; }


    uint8_t value2, mask_value2;
    Mat result_mask = Mat::zeros(bin_barcode.rows + 2, bin_barcode.cols + 2, CV_8UC1);
    Mat fill_bin_barcode = bin_barcode.clone();
    for (size_t i = 0; i < result_angle_list.size(); i++)
    {
        LineIterator line_iter(bin_barcode, result_angle_list[ i      % result_angle_list.size()],
                                            result_angle_list[(i + 1) % result_angle_list.size()]);
        for(int j = 0; j < line_iter.count; j++, ++line_iter)
        {
            value2 = bin_barcode.at<uint8_t>(line_iter.pos());
                        mask_value2 = result_mask.at<uint8_t>(line_iter.pos() + Point(1, 1));
            if (value2 == 0 && mask_value2 == 0)
            {
                floodFill(fill_bin_barcode, result_mask, line_iter.pos(), 255,
                            0, Scalar(), Scalar(), FLOODFILL_MASK_ONLY);
            }
        }
    }
    vector<Point> result_locations;
    Mat result_mask_roi = result_mask(Range(1, result_mask.rows - 1), Range(1, result_mask.cols - 1));
    
    // Mat result_result = result_mask_roi;

    findNonZero(result_mask_roi, result_locations);
    for (int i = 0; i < result_locations.size(); i++) circle(result_mask_roi, result_locations[i], 0.2, Scalar(255), -1);
    
    result_mask_roi += mask_roi_ellipse;
    imwrite("result_images/result_hull.png", result_mask_roi);


    // Mat result_draw = Mat::zeros(bin_barcode.rows , bin_barcode.cols , CV_8UC1);
    // for (int i = 0; i < result_locations.size(); i++) circle(result_draw, result_locations[i], 0.2, Scalar(255), -1);

    // namedWindow("result hull", WINDOW_NORMAL);
    // imshow("result hull", result_draw);
    // imwrite("result_hull.png", result_draw);
    // waitKey(0);
    findNonZero(result_mask_roi, result_locations);
    vector<Point> result_integer_hull;
    convexHull(result_locations, result_integer_hull);

    for (int j=0; j < result_integer_hull.size(); j++)
    {
        circle(image, result_integer_hull.at(j), 1, Scalar(248,248,97), -1);
    }
    imwrite( "result_images/result_integer_hull_locations.png", image );

    // namedWindow( "result_integer_hull locations", WINDOW_NORMAL );
    // imshow( "result_integer_hull locations", image );
    // waitKey(0);
    // for (int h = 0; h < result_angle_list.size(); h++) putText(image, to_string(h), result_angle_list[h], 1, 1, Scalar(255), 1);
    
    
    Point left_up_point, right_up_point, left_down_point, right_down_point;
    size_t r_a_size;
    // for(int i = 0; i < result_angle_list.size(); i++)
    // {
    //     if ((result_angle_list[i%3].y < result_angle_list[(i+3)%3].y) && 
    //         (result_angle_list[(i+1)%3].y < result_angle_list[(i+2)%3].y))  
    //         {
    //             left_up_point = result_angle_list[i%r_a_size];
    //             right_up_point = result_angle_list[(i+1)%r_a_size];
    //             left_down_point = result_angle_list[(i+3)%r_a_size];
    //             right_down_point = result_angle_list[(i+2)%r_a_size];
    //         }

    // }

    // if ((result_angle_list[0].y < result_angle_list[3].y) && 
    //     (result_angle_list[1].y < result_angle_list[2].y))
    // {
    //     cout<<"1\n";
    //     left_up_point = result_angle_list[0];
    //     right_up_point = result_angle_list[1];
    //     left_down_point = result_angle_list[3];
    //     right_down_point = result_angle_list[2];
    // }
    // else if ((result_angle_list[0].y > result_angle_list[3].y ) && (result_angle_list[1].y > result_angle_list[2].y))
    // {
    //     cout<<"3\n";
    //     left_up_point = result_angle_list[2];
    //     right_up_point = result_angle_list[3];
    //     left_down_point = result_angle_list[1];
    //     right_down_point = result_angle_list[0];
    // }
    // else if ((result_angle_list[0].y < result_angle_list[1].y ) && (result_angle_list[3].y < result_angle_list[2].y))
    // {
    //     cout<<"2\n";
    //     left_up_point = result_angle_list[3];
    //     right_up_point = result_angle_list[0];
    //     left_down_point = result_angle_list[2];
    //     right_down_point = result_angle_list[1];
    // }
    // else if ((result_angle_list[1].y < result_angle_list[0].y) && 
    //          (result_angle_list[2].y < result_angle_list[3].y))
    // {
    //     cout<<"4\n";
    //     left_up_point = result_angle_list[1];
    //     right_up_point = result_angle_list[2];
    //     left_down_point = result_angle_list[0];
    //     right_down_point = result_angle_list[3];

    // }
    

    // Mat points_mask = Mat::zeros(bin_barcode.size(), CV_8UC1);
    // putText(image, "left_up_point", left_up_point, 1, 1, Scalar(239, 189, 0), 1);
    // putText(image, "right_up_point", right_up_point, 1, 1, Scalar(239, 189, 0), 1);
    // putText(image, "left_down_point", left_down_point, 1, 1, Scalar(239, 189, 0), 1);
    // putText(image, "right_down_point", right_down_point, 1, 1, Scalar(239, 189, 0), 1);
    // namedWindow("check_points", WINDOW_NORMAL);
    // imshow("check_points", image);
    // waitKey(0);

    


    Mat convexhull_mask = Mat::zeros(bin_barcode.rows, bin_barcode.cols, CV_8UC1);

    vector<Point> closest_points;

    for (int i = 0; i < result_angle_list.size(); i++) putText(image, to_string(i), result_angle_list[i], 1, 2, Scalar(255), 1);
    imwrite("result_images/result_angle_list.png", image);
    
    // namedWindow("result angle list", WINDOW_NORMAL);
    // imshow("result angle list", image);
    // waitKey(0);

    size_t cont_size = result_integer_hull.size();
    vector<pair<size_t, double>> cos_vector;
    vector<size_t> indexes;
    vector<size_t> idx_closest_points_in_convexhull;

    for(size_t j = 1; j < cont_size+1; j++)
    {
        double cos_vec = getCosVectors(result_integer_hull[(j-1)%cont_size], result_integer_hull[j%cont_size], result_integer_hull[(j+1)%cont_size]);
        if((norm(result_integer_hull[(j-1)%cont_size] - result_integer_hull[j%cont_size]) <= 3) &&
            norm(result_integer_hull[(j)%cont_size] - result_integer_hull[(j+1)%cont_size]) <= 3) 
        {
            bool flag = true;

            for(int m = 0; m < indexes.size(); m++)
            {
                if(abs(static_cast<int>(j%cont_size - indexes[m]%cont_size)) < 5) flag=false;
            }
            if(flag) indexes.push_back(j%cont_size);
            continue;
        }
        cos_vector.push_back(pair<size_t, double>(j%cont_size, cos_vec));
    }
    sort(cos_vector.begin(), cos_vector.end(), sortPairAsc);
    
    for (int s = 0; s < cos_vector.size(); s++)
    {
        bool flag = true;
        for(int m = 0; m < indexes.size(); m++)
        {
            if(abs(static_cast<int>(cos_vector[s].first%cont_size - indexes[m]%cont_size)) < 2) flag=false;
        }
        if(flag) indexes.push_back(cos_vector[s].first%cont_size);
        // if(indexes.size() == 4) break;
    }
    sort(indexes.begin(), indexes.end());

    // vector<Point> contour_edge_points;
    // Point left_up_point, right_up_point, left_down_point, right_down_point;
    for (int l = 0; l < indexes.size(); l++)
    {
        // circle(convexhull_image, convexhull_contours[indexes[l]], 3, Scalar(255, 0, 0), -1);
        closest_points.push_back(result_integer_hull[indexes[l]]);
        idx_closest_points_in_convexhull.push_back(indexes[l]);
        // close_points_idx.push_back(indexes[l]);
    }

    Mat closest_points_mask = Mat::zeros(bin_barcode.size(), CV_8UC1);
    // for (int i = 0; i < closest_points.size(); i++)
    // {
    //     circle(closest_points_mask, closest_points[i], 1, Scalar(0,0,255), -1);
    //     putText(closest_points_mask, to_string(i), closest_points[i], 3, 3, Scalar(255), 1);
    // }
    // namedWindow("closest_points_mask", WINDOW_NORMAL);
    // imshow("closest_points_mask", closest_points_mask);
    // waitKey(0);

    

    

    vector<Point> temp_closest_points;
    vector<size_t> temp_idx_closest_points;
    vector<double> distances_to_angle_points;
    cout<<"closest points "<<closest_points.size()<<endl;
    cout<<result_integer_hull.size()<<endl;

    double max_dist = 0;
    size_t idx_max;

    for (size_t i = 0; i < result_angle_list.size(); i++)
    {
        double min_dist = bin_barcode.cols * bin_barcode.rows;
        size_t idx_min;
        size_t idx_max_in_convexhull;


        Point tmp_angle_point = Point(cvRound(result_angle_list[i].x), cvRound(result_angle_list[i].y));
        for (size_t j = 0; j < closest_points.size(); j++)
        {
            double dist = norm(closest_points[j] - tmp_angle_point);
            if (dist < min_dist)
            { 
                min_dist = dist;
                idx_min = j;
                idx_max_in_convexhull = idx_closest_points_in_convexhull[j];
            }
        }
        temp_closest_points.push_back(closest_points[idx_min]);
        temp_idx_closest_points.push_back(idx_max_in_convexhull);
        distances_to_angle_points.push_back(min_dist);
        if(min_dist > max_dist)
        {
            max_dist = min_dist;
            idx_max = i;
        }
    }

    closest_points = temp_closest_points;
    idx_closest_points_in_convexhull = temp_idx_closest_points;

    for (int i = 0; i < closest_points.size(); i++)
    {
        circle(close_points, closest_points[i], 1, Scalar(255, 255, 255), -1);
        circle(close_points, result_angle_list[i], 1, Scalar(0, 0, 255), -1);
        Point tmp = Point(cvRound(result_angle_list[i].x), cvRound(result_angle_list[i].y));
        double dist = norm(closest_points[i] - tmp);
        cout<<"dist between close point and angle point "<<dist<<endl;
    }
    circle(close_points, closest_points[idx_max], 1, Scalar(255), -1);

    imwrite("result_images/closest_and_angle_points.png", close_points);


    bool find_fourth_point = false;
    int count = 0;
    for (int i = 0; i < distances_to_angle_points.size(); i++)
    {
        cout<<abs(max_dist - distances_to_angle_points[i])<<endl;
        if (abs(max_dist - distances_to_angle_points[i]) < 10) count++;
    }
    cout<<"count "<<count<<endl;
    if (count == 1) 
    {
        find_fourth_point = true;
    }
    cout<<"find_fourth_point "<<find_fourth_point<<endl;

    Mat clone_image = image.clone();
    cout<<"result integer hull size "<<result_integer_hull.size()<<endl;
    int min_size = result_integer_hull.size();
    vector<vector<Point>> sides_points;
    int min_index_1, min_index_2;
    for(int i = 0; i < idx_closest_points_in_convexhull.size(); i++)
    {
        vector<Point> points;
        int start = idx_closest_points_in_convexhull[i], end = idx_closest_points_in_convexhull[(i+1)%idx_closest_points_in_convexhull.size()];
        
        if (start < end)
        {
            points.insert(points.end(), result_integer_hull.begin() + start, result_integer_hull.begin() + end + 1);

        }
        else
        {
            points.insert(points.end(), result_integer_hull.begin() + start, result_integer_hull.end());
            points.insert(points.end(), result_integer_hull.begin(), result_integer_hull.begin() + end + 1);
        }
        if (abs(result_integer_hull[start].x - result_integer_hull[end].x) > abs(result_integer_hull[start].y - result_integer_hull[end].y))
        {
            sort(points.begin(), points.end(), sortPointsByX);
        }
        else
        {
            sort(points.begin(), points.end(), sortPointsByY);
        }
        sides_points.push_back(points);
        printf("points between %d and %d = %d\n", start, end, (int) points.size());
    }
    // for (int i = 0; i < sides_points.size(); i++)
    // {
    //     cout<<"i "<<i<<endl;
    //     cout<<sides_points[i].front()<<endl;
    //     for(int j = 0; j < sides_points[i].size(); j++)
    //     {
    //         cout<<sides_points[i][j]<<endl;
    //     }
    // }

    

    if(find_fourth_point)
    {
        Point a1, a2, b1, b2;
        size_t side_size = sides_points.size();
        Point temp_point = closest_points[idx_max];
        // vector<Point> &current_side, &next_side;
        vector<Point> current_side_points, next_side_points;
        int start_current, end_current, step_current, start_next, end_next, step_next;
        vector<Point>::iterator it_a, it_b;

        
        vector<Point> &current_side = sides_points[(idx_max+3)%4];
        vector<Point> &next_side = sides_points[idx_max];
        
        float dist_to_current_side = abs(distancePt(temp_point, result_angle_list[(idx_max+3)%4], result_angle_list[idx_max]));
        float dist_to_next_side = abs(distancePt(temp_point, result_angle_list[idx_max], result_angle_list[(idx_max+1)%4]));
        cout<<"dist_to_current_side "<<dist_to_current_side<<endl;
        cout<<"dist_to_next_side "<<dist_to_next_side<<endl;
// circle(image, sides_points[3].back(), 5, Scalar(255,255,255), -1);
        // if (dist_to_current_side > dist_to_next_side)
        // {
        //     swap(current_side, next_side);
        // }
cout<<"current_side\n";
        for (int i = 0; i < current_side.size(); i++)
        {
            cout<<current_side[i]<<endl;
        }
// circle(image, sides_points[1].front(), 5, Scalar(255,255,255), -1);
        bool add_point_4 = true;
        

        if(temp_point == current_side.front())
        {
            start_current = 0;
            end_current = current_side.size() - 2;
            step_current = 1;
            it_a = current_side.begin();
        }
        else if(temp_point == current_side.back())
        {
            start_current = current_side.size() - 1;
            end_current = 1;
            step_current = -1;
            it_a = current_side.end()-1;
        } 
        if(temp_point == next_side.front())
        {
            start_next = 0;
            end_next = next_side.size() - 2;
            step_next = 1;
            it_b = next_side.begin();
        }
        else if(temp_point == next_side.back())
        {
            start_next = next_side.size() - 1;
            end_next = 1;
            step_next = -1;
            it_b = next_side.end()-1;
        }


        current_side_points = get_points(image, current_side, start_current, end_current, step_current);
        next_side_points = get_points(image, next_side, start_next, end_next, step_next);


        a1 = current_side_points[0];
        a2 = current_side_points[1];

        b1 = next_side_points[0];
        b2 = next_side_points[1];

        cout<<"a1 b1 "<<norm(a1 - b1)<<endl;
        // cout<<"next side "<<next_side_points.size()<<endl;

        if(norm(a1 - b1)<10 && next_side_points.size() > 2)
        {
            cout<<"krya\n";
            b1 = next_side_points[1];
            b2 = next_side_points[2];
        }

        circle(image, a1, 1, Scalar(0, 255, 255), -1);
        // putText(image, "a1", a1, 1, 1, Scalar(145, 34, 200), 1);
        circle(image, a2, 1, Scalar(0, 0, 255), -1);
        // putText(image, "a2", a2, 1, 1, Scalar(145, 34, 200), 1);
        circle(image, b1, 1, Scalar(0, 255, 255), -1);
        // putText(image, "b1", b1, 1, 1, Scalar(145, 34, 200), 1);
        circle(image, b2, 1, Scalar(0, 255, 255), -1);
        // putText(image, "b2", b2, 1, 1, Scalar(145, 34, 200), 1);

        Point point_4 = intersectionLines(a1, a2, b1, b2);
        circle(image, point_4, 1, Scalar(145, 34, 200), -1);
        imwrite("result_images/4_point.png", image);
// exit(0);

        cout<<"current_side "<<current_side.size()<<endl;
        for(int i = 0; i < current_side.size(); i++)
        {
            cout<<current_side[i]<<endl;
        }
        // cout<<"a1 "<<a1<<endl;
        // cout<<"*it_a "<<*it_a<<endl;
        // it_a = current_side.erase(it_a);
        // // it_a += step_current;
        // // // it_a = temp_it_a;
        // cout<<"*it_a "<<*it_a<<endl;
        // cout<<"end "<<*current_side.end()<<endl;
        // if(it_a == current_side.end()) cout<<"kkkk\n";
        // // it_a = current_side.erase(it_a);
        // // cout<<"*it_a "<<*it_a<<endl;
        cout<<"next_side "<<next_side.size()<<endl;
        for(int i = 0; i < next_side.size(); i++)
        {
            cout<<next_side[i]<<endl;
        }
        // cout<<"b1 "<<b1<<endl;
        // cout<<"*it_b "<<*it_b<<endl;
        // it_b = next_side.erase(it_b);
        // cout<<"*it_b "<<*it_b<<endl;
        // if(*it_b == b1) cout<<"*it_b == b1\n";



        while (*it_a != a1)
        {
            cout<<"removed "<<*it_a<<endl;
            it_a = current_side.erase(it_a);
            if (it_a == current_side.end())
            {
                it_a += step_current;
            }
        }
        while (*it_b != b1)
        {
            cout<<"removed "<<*it_b<<endl;
            it_b = next_side.erase(it_b);
            if (it_b == next_side.end())
            {
                it_b += step_next;
            }
        }


                
        for (int k = 0; k < result_integer_hull.size(); k++)
        {
            if(point_4 == result_integer_hull[k]) 
            {
                add_point_4 = false;
                break;
            }
        }
        // cout<<"add_point_4 "<<add_point_4<<endl;
        for (int k = 0; k < result_angle_list.size(); k++)
        {
            Point angle_point = Point(cvRound(result_angle_list[k].x), cvRound(result_angle_list[k].y));
            if(point_4 == angle_point) 
            {
                add_point_4 = false;
                current_side.emplace(it_a, angle_point);
                next_side.emplace(it_b, angle_point);
                closest_points[idx_max] = angle_point;
                break;
            }

        }
        cout<<"add_point_4 "<<add_point_4<<endl;
        
        cout<<"closest_point before "<<closest_points[idx_max]<<endl;
        if(add_point_4)
        {
            current_side.emplace(it_a, point_4);
            next_side.emplace(it_b, point_4);
            closest_points[idx_max] = point_4;
        }
        cout<<"point_4 "<<point_4<<endl;
        cout<<"closest_point after "<<closest_points[idx_max]<<endl;

        for (int i = 0; i < current_side.size(); i++)
        {
            circle(image, current_side[i], 1, Scalar(0,255,0), -1);
        }
        for (int i = 0; i < next_side.size(); i++)
        {
            circle(image, next_side[i], 1, Scalar(0,255,0), -1);
        }
        imwrite("result_images/4_point.png", image);

    }
    // exit(0);
    // namedWindow("sides_points_mask", WINDOW_NORMAL);
    Mat sides_points_mask = Mat::zeros(bin_barcode.size(), CV_8UC1);

    for (int i = 0; i < closest_points.size(); i++)
    {
        circle(sides_points_mask, closest_points[i], 3, Scalar(255), -1);
        // putText(closest_points_mask, to_string(i), closest_points[i], 3, 3, Scalar(255), 1);
    }
    imwrite("result_images/closest_points_mask.png", sides_points_mask);

    // namedWindow("closest_points_mask", WINDOW_NORMAL);
    // imshow("closest_points_mask", sides_points_mask);
    // waitKey(0);


    for (int i = 0; i < sides_points.size(); i++)
    {
        for (int j = 0; j < sides_points[i].size(); j++ )
        {
            circle(sides_points_mask, sides_points[i][j], 1, Scalar(255), -1);
        }
        // for (int j = 0; j < sides_points[i].size(); j++ )
        // {
        //     putText(sides_points_mask, to_string(j), sides_points[i][j], 3, 1, Scalar(255), 1);
        // }
        // imshow("sides_points_mask", sides_points_mask);
        // waitKey(0);
    }






//достроить выпуклую оболочку

// Mat hull_contours_mask = Mat::zeros(bin_barcode.rows, bin_barcode.cols, CV_8UC1);
// for (int i = 0; i < hull_points.size(); i++)
// {
//     putText(hull_contours_mask, to_string(i), hull_points[i][0], 2, 2, Scalar(255), 1);
//     for (int j = 0; j < hull_points[i].size(); j++)
//     {
//         circle(hull_contours_mask, hull_points[i].at(j), 1, Scalar(255), -1);
//         // putText(hull_contours_mask, to_string(j), hull_points[i].at(j), 1, 1, Scalar(255), 1);
//     }
// }
// namedWindow("hull_contours_mask", WINDOW_NORMAL);
// imshow("hull_contours_mask", hull_contours_mask);
// waitKey(0);

//выпуклая оболочка контуров
    // namedWindow("convexhull_contours", WINDOW_NORMAL);

    vector<Point> side;
    vector<vector<Point>> markers_edge_points;
    for(int i = 0; i < hull_points.size(); i++)
    {
        vector<Point> convexhull_contours, new_convexhull_contours;
        convexHull(hull_points[i], convexhull_contours);
        cout<<hull_points[i].size()<<endl;
        for(int j = 0; j < convexhull_contours.size(); j++)
        {
            circle(convexhull_image, convexhull_contours[j], 1, Scalar(164, 56, 154), -1);
        }
        // imshow("convexhull_contours", convexhull_image);
        // waitKey(0);

        size_t cont_size = convexhull_contours.size();
        vector<pair<size_t, double>> cos_vector;
        vector<size_t> indexes;

        for(size_t j = 1; j<cont_size+1; j++)
        {
            double cos_vec = getCosVectors(convexhull_contours[(j-1)%cont_size], convexhull_contours[j%cont_size], convexhull_contours[(j+1)%cont_size]);
            if((norm(convexhull_contours[(j-1)%cont_size] - convexhull_contours[j%cont_size]) <= 3) &&
                norm(convexhull_contours[(j)%cont_size] - convexhull_contours[(j+1)%cont_size]) <= 3) 
            {
                bool flag = true;

                for(int m = 0; m < indexes.size(); m++)
                {
                    if(abs(static_cast<int>(j%cont_size - indexes[m]%cont_size)) < 5) flag=false;
                }
                if(flag) indexes.push_back(j%cont_size);
                continue;
            }
            cos_vector.push_back(pair<size_t, double>(j, cos_vec));
        }
        sort(cos_vector.begin(), cos_vector.end(), sortPairAsc);
      
        for (int s = 0; s < cos_vector.size(); s++)
        {
            bool flag = true;
            for(int m = 0; m < indexes.size(); m++)
            {
                if(abs(static_cast<int>(cos_vector[s].first%cont_size - indexes[m]%cont_size)) < 2) flag=false;
            }
            if(flag) indexes.push_back(cos_vector[s].first%cont_size);
            if(indexes.size() == 4) break;
        }
        sort(indexes.begin(), indexes.end());

        vector<Point> contour_edge_points;
        Point left_up_point, right_up_point, left_down_point, right_down_point;
        for (int l = 0; l < indexes.size(); l++)
        {
            circle(convexhull_image, convexhull_contours[indexes[l]], 1, Scalar(255, 0, 0), -1);
            contour_edge_points.push_back(convexhull_contours[indexes[l]]);
        }
        markers_edge_points.push_back(contour_edge_points);

        

        // imshow("convexhull_contours", convexhull_image);
        // waitKey(0);
    }
    imwrite("result_images/convexhull_contours.png", convexhull_image);



//end выпуклая оболочка контуров


// точки маркеров
 
    // namedWindow("close points", WINDOW_NORMAL);

    // vector<pair<int, vector<Point>>> cur_sides, marker_points
    int id_cur_1, id_cur_2;
    vector<int> cur_indexes(2);
    // size_t min_point_count = result_integer_hull.size();
    // for (size_t i = 0; i < sides_points.size(); i++)
    // {
    //     if (sides_points[i].size() < min_point_count)
    //     {
    //         min_point_count = sides_points[i].size();
    //         id_cur_1 = i;
    //         cur_indexes.at(0) = i;
    //         cur_indexes.at(1) = (i+2)%sides_points.size();
    //         id_cur_2 = (i+2)%sides_points.size();
    //     }
    // }

    double max_dist_to_arc_side = 0.0;
    for (int i = 0; i < closest_points.size(); i++)
    {
        putText(add_points_image, to_string(i), closest_points[i], 1, 1, Scalar(0,255,255), 1);
    }
    for (int k = 0 ;k < sides_points.size(); k++)
    {
        putText(add_points_image, to_string(k), sides_points[k].front(), 2, 2, Scalar(0,0,255), 1);
        
    }
    for (int i = 0; i < closest_points.size(); i++)
    {
        double dist_to_arc = 0.0;
        Point arc_begin = closest_points[i];
        Point arc_end = closest_points[(i+1)%4];
        for (int j = 1; j < sides_points[i].size()-1; j++)
        {
            Point arc_point = sides_points[i][j];
            
            // Point perpendicular = pointProjection(arc_begin, arc_end, arc_point);
            // line(add_points_image, arc_point, perpendicular, Scalar(100, 250, 0), 1);
            circle(add_points_image, arc_point, 1, Scalar(255), -1);

            double dist = abs(distancePt(arc_point, arc_begin, arc_end));
            dist_to_arc += dist;

        }
        cout<<"dist_to_arc "<<dist_to_arc<<endl;
        cout<<sides_points[i].size()-2<<endl;

        dist_to_arc /= (sides_points[i].size()-2);
        cout<<"mean dist_to_arc "<<dist_to_arc<<endl;

        if (dist_to_arc > max_dist_to_arc_side)
        {
            max_dist_to_arc_side = dist_to_arc;
            id_cur_1 = i;
            id_cur_2 = (i+2)%4;
        }

    }
    imwrite("result_images/points.png", add_points_image);
    cout<<"id_cur_1 "<<id_cur_1<<" id_cur_2 "<<id_cur_2<<endl;
    cur_indexes.at(0) = id_cur_1;
    cur_indexes.at(1) = id_cur_2;



    vector<int> cur_add_indexes;
    for (int i = 0; i < cur_indexes.size(); i++)
    {
        size_t idx_side = cur_indexes.at(i);
        double max_dist = norm(closest_points.at(idx_side) - closest_points.at((idx_side+1)%4));
        double real_max_dist = 0;
        size_t side_size = sides_points.at(idx_side).size();
        for (size_t j = 0; j < side_size-1; j++)
        {
            double dist = norm(sides_points.at(idx_side).at(j) - sides_points.at(idx_side).at(j+1));
            if (dist > real_max_dist)
            {
                real_max_dist = dist;
            }
        }
        cout<<"real_max_dist "<<real_max_dist<<endl;
        cout<<"max_dist "<<max_dist<<endl;
        cout<<(1.0/2.0) * max_dist<<endl;
        if (real_max_dist > ((1.0/2.0) * max_dist))
        {
            cur_add_indexes.push_back(cur_indexes.at(i));
        }

    }


    
    cout<<"cur_indexes\n" ;
    for(int i = 0 ; i < cur_indexes.size(); i++)
    {
        cout<<cur_indexes[i]<<endl;
    }
    cout<<"cur_add_indexes\n" ;
    for(int i = 0 ; i < cur_add_indexes.size(); i++)
    {
        cout<<cur_add_indexes[i]<<endl;
    }
// exit(0);

    vector<vector<Point>> sides_add_points;
    vector<pair<int,vector<Point>>> marker_points;
    bool make_full_side_first = false;
    for (int i = 0; i < cur_add_indexes.size(); i++)
    {
        int idx = cur_add_indexes[i];
        vector<size_t> indexes;
        for (size_t j = 0; j < markers_edge_points.size(); j++)
        {
            for (size_t k = 0; k < markers_edge_points[j].size(); k++)
            {
                if (norm(Point(cvRound(result_angle_list[idx].x), cvRound(result_angle_list[idx].y)) - markers_edge_points[j][k]) < 5) 
                {
                    indexes.push_back(j);
                }
                if (norm(Point(cvRound(result_angle_list[(idx+1)%4].x), cvRound(result_angle_list[(idx+1)%4].y)) - markers_edge_points[j][k]) < 5) 
                {
                    indexes.push_back(j); 
                }
            }
        }
        cout<<"indexes size "<<indexes.size()<<endl;
        if (i == 0)
        {
            if (indexes.size() == 1)
            {
                make_full_side_first = true;
            }
        }
        for (int h = 0; h < indexes.size(); h++)
        {
            vector<pair<size_t, double>> id_dist;
            vector<Point> points;
            for (size_t k = 0; k < markers_edge_points[indexes[h]].size(); k++)
            {
                double dist;
                dist = norm(markers_edge_points[indexes[h]][k] - Point(cvRound(result_angle_list[idx].x), cvRound(result_angle_list[idx].y)));
                dist += norm(markers_edge_points[indexes[h]][k] - Point(cvRound(result_angle_list[(idx+1)%4].x), cvRound(result_angle_list[(idx+1)%4].y)));
                dist /= 2;
                id_dist.push_back(pair<size_t, double>(k, dist));

            }
            if(!id_dist.empty())

            {

                sort(id_dist.begin(), id_dist.end(), sortPairDesc);
                cout<<"marker "<<h<<endl;
                for (int s = 0; s < id_dist.size(); s++)
                {
                    cout << id_dist[s].first << ": " << id_dist[s].second << endl;
                    putText(image, to_string(id_dist[s].first), markers_edge_points[indexes[h]][id_dist[s].first], 1, 1, Scalar(0,0,255), 1);
                }
                Point p1, p2;
                int index_p1, index_p2;
                for (int r = 4; r > 0; r--)
                {
                    if((id_dist[0].first == r%4) && (id_dist[1].first == (r-1)%4))
                    {
                        index_p1 = id_dist[0].first;
                        index_p2 = id_dist[1].first;
                        
                    }
                    else if((id_dist[1].first == r%4) && (id_dist[0].first == (r-1)%4))
                    {
                        index_p1 = id_dist[1].first;
                        index_p2 = id_dist[0].first;
                        
                    }
                }
                
                p1 = markers_edge_points[indexes[h]][index_p1];
                p2 = markers_edge_points[indexes[h]][index_p2];
                putText(image, "p1", p1, 1, 2, Scalar(255), 1);
                putText(image, "p2", p2, 1, 2, Scalar(255), 1);

                int index1, index2;
                for(int n = 0; n < hull_points[indexes[h]].size(); n++)
                {
                    
                    if (hull_points[indexes[h]][n] == p1)
                    {
                        index1 = n;
                    }
                    if (hull_points[indexes[h]][n] == p2)
                    {
                        index2 = n;
                    }
                }

                if (index1 > index2)
                {

                    for (int l = index1; l < hull_points[indexes[h]].size(); l++)
                    {
                        // circle(image, hull_points[indexes[h]][l], 3, Scalar(0,255,0), -1);
                        // putText(image, to_string(l), hull_points[indexes[h]][l], 1, 1, Scalar(29, 50, 138), 1);
                        points.push_back(hull_points[indexes[h]][l]);
                    }
                    for (int l = 0; l <= index2; l++)
                    {
                        // circle(image, hull_points[indexes[h]][l], 3, Scalar(0,255,0), -1);
                        // putText(image, to_string(l), hull_points[indexes[h]][l], 1, 1, Scalar(29, 50, 138), 1);

                        points.push_back(hull_points[indexes[h]][l]);
                    
                    }

                }
                else
                {
                    for (int l = index1; l <= index2; l++)
                    {
                    //     circle(image, hull_points[indexes[h]][l], 3, Scalar(0,255,0), -1);
                    //     putText(image, to_string(l), hull_points[indexes[h]][l], 1, 1, Scalar(29, 50, 138), 1);

                        points.push_back(hull_points[indexes[h]][l]);
                    
                    }
                }
                // points.push_back(p2);
                if (abs(p1.x - p2.x) > abs(p1.y - p2.y))
                {
                    sort(points.begin(), points.end(), sortPointsByX);
                }
                else
                {
                    sort(points.begin(), points.end(), sortPointsByY);
                }

            }
            cout<<"points size "<<points.size()<<endl;
            marker_points.push_back(pair<int,vector<Point>>(idx,points));

            printf("idx %d points size %d\n", idx, points.size());
            for (int p = 0; p < points.size(); p++)
                circle(add_points_image, points[p], 2, Scalar(100, 250, 0), -1);

        }

        

        // for (int p = 0; p < points.size(); p+=15)
        //     putText(image, to_string(p), points[p], 1, 1, Scalar(29, 50, 138), 1);
        // imshow("close points", image);
        // waitKey(0);


        // namedWindow("lines", WINDOW_NORMAL);
        // imshow("lines", image);
        // waitKey(0);

        // sides_add_points.push_back(pointss);


    }
    
    // if (make_full_side_first)
    // {

    //     reverse(marker_points.begin(), marker_points.end());
    //     reverse(cur_indexes.begin(), cur_indexes.end());
    //     reverse(new_cur_indexes.begin(), new_cur_indexes.end());
    // }
    imwrite("result_images/points.png", add_points_image);
    // for(int i = 0; i < marker_points[0].size(); i++)
    // {
    //     circle(spline_image, marker_points[0][i], 1, Scalar(0, 0, 255), -1);

    // }
    // imwrite("result_images/marker_points.png", spline_image);

    size_t number_of_cur_sides = cur_indexes.size();
    map<int,vector<Point>> temp_sides_add_points;
    float mean_step = 0.0f;
    // vector<float> steps;
    for(int i = 0; i < marker_points.size(); i++)
    {
        int number_of_points = 3;
        int idx_side = marker_points[i].first;
        size_t size = marker_points[i].second.size();
        cout<<i<<" marker points size = "<<size<<endl;

        float step = static_cast<float>(size) / number_of_points;
        mean_step += step;
        vector<Point> temp_points;
        for (int j = 0; j < number_of_points; j++) 
        {
            float val = j * step;
            int idx = cvRound(val) >= size ? size - 1 : cvRound(val);
            // cout << j << " " << idx << " " << temp_points.size() << endl;
            // cout<<"add to temp points "<<marker_points[i].second[idx]<<endl;
            temp_points.push_back(marker_points[i].second[idx]);
        }
        temp_points.push_back(marker_points[i].second.back());
        // cout<<"idx_side "<<idx_side<<endl;
        // cout<<"temp_points\n"<<temp_points<<endl;
        if(temp_sides_add_points.count(idx_side) == 1)
        {
            // cout<<"krya "<<idx_side<<endl;
            temp_sides_add_points[idx_side].insert(temp_sides_add_points[idx_side].end(),
                                                   temp_points.begin(), temp_points.end());
        }
        temp_sides_add_points.insert(pair<int,vector<Point>>(idx_side, temp_points));
        // cout<<"temp_sides_add_points size "<<temp_sides_add_points.size()<<endl;

        // if ((i+1)%2 == 0)
        // {
        //     steps.push_back(mean_step);
        //     mean_step = 0;
        // }
        
    }
    cout<<"mean_step "<<mean_step<<endl;
    cout<<"marker_points.size() "<<marker_points.size()<<endl;
    if (marker_points.size() > 0)
    {
        mean_step /= marker_points.size();
    }
    cout<<"temp_sides_add_points size "<<temp_sides_add_points.size()<<endl;
    // exit(0);
// for (int i = 0; i < temp_sides_add_points[1].size(); i++)
// {
//     circle(spline_image, temp_sides_add_points[1][i], 3, Scalar(0,255,255), -1);
// }
// imwrite("result_images/temp_sides_add_points.png", spline_image);
    cout<<"temp_sides_add_points size "<<temp_sides_add_points.size()<<endl;
// exit(0);
    // for (int i = 0; i < sides_points.size(); i++)
    // {
    //     for (int j = 0; j < sides_points[i].size(); j++)
    //     {
    //         for (int k = 0; k < sides_add_points[i].size(); k++)
    //         {
    //             if (sides_points[i][j] != sides_add_points[i][k])
    //             {
    //                 sides_add_points[i].push_back(sides_points[i][j]);
    //                 break;
    //             }
    //         }
    //     }
    // }

 
 // end точки маркеров

//аппроксимировать кривую


// разбивать на точки для сплайна по маркерам


    // size_t cur_side_size = sides_add_points[id_cur_1].size();
    // double max_dist_in_side = cur_side_size;
    // for (int i = 0; i < cur_side_size-1; i++) 
    // {
    //     double dist = norm(sides_add_points[id_cur_1][i] - sides_add_points[id_cur_1][i+1]);
    //     if (dist > max_dist_in_side)
    //     {
    //         max_dist_in_side = dist;
    //     }
    // }
    // double param = max_dist_in_side/cur_side_size;
    // int step = cvRound(sqrt(cur_side_size)) + cvRound(sqrt(param));
    // // int step = 8;
    // cout<<"sides_add_points[id_cur_1][i]\n";
    // for (int i = 0; i < sides_add_points[id_cur_1].size(); i++)
    // {
    //     circle(add_points_image, sides_add_points[id_cur_1][i], 3, Scalar(0, 200, 0), -1);

    // }

    
    // for (int i = 0; i < sides_add_points[id_cur_1].size(); i+=step) // подобрать количество точек
    // {
    //     temp_sides_add_points.push_back(sides_add_points[id_cur_1][i]);
    // }
    // int idx = 1;
    // // int j = 1;
    // for(int i = 0; i < sides_points[idx].size(); i++)
    // {
    //     circle(spline_image, sides_points[idx][i], 1, Scalar(255, 255, 255), -1);

    // }
    // for(int i = 0; i < temp_sides_add_points[idx].size(); i++)
    // {
    //     circle(spline_image, temp_sides_add_points[idx][i], 1, Scalar(0, 0, 255), -1);

    // }
    // imwrite("result_images/check_points.png", spline_image);

cout<<"step mean "<<mean_step<<endl;
cout<<"cur indexes size "<<cur_indexes.size()<<endl;
cout<<"temp_sides_add_points size "<<temp_sides_add_points.size()<<endl;
if (temp_sides_add_points.size() > 0)
{
    for (int i = 0; i < cur_add_indexes.size(); i++)
    {
        cout<<"i "<<i<<endl;
        int idx = cur_add_indexes[i];
        cout<<"idx "<<idx<<endl;

        vector<int> sides_points_indexes;

        for (int j = 0; j < sides_points[idx].size(); j++)
        {
            cout<<"j "<<j<<endl;

            bool flag = true;
            for (int k = 0; k < temp_sides_add_points[idx].size(); k++)
            {
                // cout<<norm(sides_points[idx][j] - temp_sides_add_points[i][k])<<endl;
                double dist = norm(sides_points[idx][j] - temp_sides_add_points[idx][k]);
                cout<<k<<" "<<dist<<endl;
                if (dist < mean_step)
                {
                    flag = false;
                    break;
                }
            }
            if (flag)
            {
                sides_points_indexes.push_back(j);
            }
        }

        for (int j = 0; j < sides_points_indexes.size(); j++)
        {
            cout<<"krya\n";
            temp_sides_add_points[idx].push_back(sides_points[idx][sides_points_indexes[j]]);
            circle(spline_image, sides_points[idx][sides_points_indexes[j]], 1, Scalar(255), -1);
            putText(spline_image, to_string(j), sides_points[idx][sides_points_indexes[j]], 1, 1, Scalar(255), 1);
        }


    }
}
    imwrite("result_images/order_add_points.png", spline_image);

    // exit(0);
    cout<<"add points size "<<temp_sides_add_points.size()<<endl;
    // exit(0);
    if (temp_sides_add_points.size() < 2)
    {
        for (int i = 0; i < cur_indexes.size(); i++)
        {
            if(temp_sides_add_points.count(cur_indexes[i]) == 0)
            {
        
                int idx_second_cur_side = cur_indexes[i];
                temp_sides_add_points.insert(pair<int,vector<Point>>(idx_second_cur_side, sides_points[idx_second_cur_side]));
            }
        }
    }
    // cout<<"add points size "<<temp_sides_add_points.size()<<endl;
    for (std::map<int,vector<Point>>::iterator it=temp_sides_add_points.begin(); it!=temp_sides_add_points.end(); ++it)
    // for (int i = 0; i < temp_sides_add_points.size(); i++)
    {
        cout<<"idx side in temp add points "<<it->first<<endl;
        cout<<"temp_sides_add_points[i].size() "<<it->second.size()<<endl;
        for (int j = 0; j < it->second.size(); j++)
        {
            cout<<it->second[j]<<endl;
            circle(add_points_image, it->second[j], 3, Scalar(250, 100, 0), -1);
 
        }
    }
    imwrite("result_images/spline_points.png", add_points_image);

    for (std::map<int,vector<Point>>::iterator it=temp_sides_add_points.begin(); it!=temp_sides_add_points.end(); ++it)
    // for (int i = 0; i < temp_sides_add_points.size(); i++)
    {
        Point p1 = it->second.front();
        Point p2 = it->second.back();
        if (abs(p1.x - p2.x) > abs(p1.y - p2.y))
        {
            cout<<"by X\n";
            sort(it->second.begin(), it->second.end(), sortPointsByX);
        }
        else
        {
            cout<<"by Y\n";
            sort(it->second.begin(), it->second.end(), sortPointsByY);
        }
    }
cout<<"krya krya\n";
    for (std::map<int,vector<Point>>::iterator it=temp_sides_add_points.begin(); it!=temp_sides_add_points.end(); ++it)
    // for (int i = 0; i < temp_sides_add_points.size(); i++)
    {
        cout<<"idx side in temp add points "<<it->first<<endl;
        cout<<"temp_sides_add_points[i].size() "<<it->second.size()<<endl;
        for (int j = 0; j < it->second.size(); j++)
        {
            cout<<it->second[j]<<endl;
            // circle(add_points_image, it->second[j], 3, Scalar(250, 100, 0), -1);

        }
    }

    // namedWindow("order add points", WINDOW_NORMAL);
    // imshow("order add points", spline_image);
    // waitKey(0);
    // vector<vector<int>> x_arr(temp_sides_add_points.size()), y_arr(temp_sides_add_points.size());
    // for (int i = 0; i < temp_sides_add_points.size(); i++) // подобрать количество точек
    // {
    //     cout<<"krya krya "<<i<<endl;
    //     cout<<temp_sides_add_points[i].size()<<endl;

    //     for (int j = 0; j < temp_sides_add_points[i].size(); j++) // подобрать количество точек
        
    //     {
    //         x_arr[i].push_back(cvRound(temp_sides_add_points[i][j].x));
    //         y_arr[i].push_back(cvRound(temp_sides_add_points[i][j].y));
    //     }
    // }
cout<<"krya krya\n";

    // circle(spline_image, sides_add_points[id_cur_1].back(), 1, Scalar(0,0,255), -1);
    // namedWindow("back point", WINDOW_NORMAL);
    // imshow("back point", spline_image);
// waitKey(0);
    // if (sides_add_points[id_cur_1].back() != Point(x_arr.back(), y_arr.back()))
    // {
    //     x_arr.push_back(sides_add_points[id_cur_1].back().x);
    //     y_arr.push_back(sides_add_points[id_cur_1].back().y);
    // }

    // step = cvRound(sqrt(sides_points[id_cur_1].size()));
    // cout<<"sides_add_points[id_cur_1][i]\n";
    // for (int i = 0; i < sides_points[id_cur_1].size(); i++) // подобрать количество точек
    // {
    //     x_arr.push_back(cvRound(sides_points[id_cur_1][i].x));
    //     y_arr.push_back(cvRound(sides_points[id_cur_1][i].y));
    // }
    // for (int i = 0; i < sides_points.size(); i++)
    // {
    //     for (int j = 0; j < sides_points[i].size(); j++)
    //     {
    //         for (int k = 0; k < sides_add_points[i].size(); k++)
    //         {
    //             if (sides_points[i][j] != sides_add_points[i][k])
    //             {
    //                 sides_add_points[i].push_back(sides_points[i][j]);
    //                 break;
    //             }
    //         }
    //     }
    // }
    // double mean_dist = 0;
    // for (int i = 0; i < sides_add_points[id_cur_1].size()-1; i++) // подобрать количество точек
    // {
    //     double d = norm(sides_add_points[id_cur_1][i] - sides_add_points[id_cur_1][i+1]);
    //     cout<<sides_add_points[id_cur_1][i]<<endl;
    //     cout<<d<<endl;
    //     circle(spline_image, sides_add_points[id_cur_1][i+1], 3, Scalar(0,0,255), -1);
    //     putText(spline_image, to_string(d), sides_add_points[id_cur_1][i+1], 1, 1, Scalar(255), 1);
    //     // mean_dist += norm(sides_add_points[id_cur_1][i] - sides_add_points[id_cur_1][i+16]);
    // }
    // mean_dist /= (x_arr.size() - 1);

    // cout<<x_arr.size()<<" "<<y_arr.size()<<endl;

    vector<Point> left_points_to_cut;
    int start, end;
    vector<vector<double>> S;

    vector<vector<Point>> points_to_cut(cur_indexes.size());
    for (int idx = 0; idx < cur_indexes.size(); idx++)    
    {
        id_cur_1 = cur_indexes[idx];

        vector<Point> spline_points =  temp_sides_add_points.find(id_cur_1)->second;
        vector<int> x_arr, y_arr;

        for (int j = 0; j < spline_points.size(); j++) // подобрать количество точек
        
        {
            x_arr.push_back(cvRound(spline_points[j].x));
            y_arr.push_back(cvRound(spline_points[j].y));
        }


        if (abs(x_arr.front() - x_arr.back()) > abs(y_arr.front() - y_arr.back()))
        {
            cout<<"horizontal\n";
            S = spline(y_arr, x_arr);
            if(closest_points[id_cur_1].x > closest_points[(id_cur_1+1)%4].x)
            {
                start = (id_cur_1+1)%4;
                end = id_cur_1;
            }
            else
            {
                start = id_cur_1;
                end = (id_cur_1+1)%4;
            }
            if (idx == 0 )
            {
                circle(spline_image, closest_points[start], 3, Scalar(0,0,255), -1);
                circle(spline_image, closest_points[end], 3, Scalar(0,0,255), -1);
                circle(spline_image, Point(x_arr[0], y_arr[0]), 2, Scalar(0,255,0), -1);

            }
            
            for (int x = closest_points[start].x; x <= closest_points[end].x; x++)
            {
                for (int i = 0; i < x_arr.size()-1; i++)
                {
                    if ((x >= x_arr.at(i)) && (x < x_arr.at(i+1)))
                    {
                        
                        double y = S.at(i).at(0) + S.at(i).at(1) * (x - x_arr.at(i)) + S.at(i).at(2) * pow((x - x_arr.at(i)), 2)
                                                                                    + S.at(i).at(3) * pow((x - x_arr.at(i)), 3);
                        circle(spline_image, Point(cvRound(x), y), 1, Scalar(255), -1);
                        points_to_cut[idx].push_back(Point(cvRound(x), y));
                    }
                }
            }
        }
        else
        {
            cout<<"vertical\n";
            S = spline(x_arr, y_arr);
            cout<<"S size "<<S.size()<<endl;
            cout<<"y arr "<<y_arr.size()<<endl;
            if(closest_points[id_cur_1].y > closest_points[(id_cur_1+1)%4].y)
            {
                start = (id_cur_1+1)%4;
                end = id_cur_1;
            }
            else
            {
                start = id_cur_1;
                end = (id_cur_1+1)%4;
            }
            if (idx == 1 )
            {
                circle(spline_image, closest_points[start], 3, Scalar(0,0,255), -1);
                circle(spline_image, closest_points[end], 3, Scalar(0,0,255), -1);
                circle(spline_image, Point(x_arr[0], y_arr[0]), 2, Scalar(0,255,0), -1);

            }
            circle(spline_image, closest_points[end], 1, Scalar(0,0,255), -1);
            cout<<closest_points[end]<<endl;
            for (int y = closest_points[start].y; y <= closest_points[end].y; y++)
            {
                for (int i = 0; i < y_arr.size()-1; i++)
                {
                    if ((y >= y_arr.at(i)) && (y < y_arr.at(i+1)))
                    {
                        double x = S.at(i).at(0) + S.at(i).at(1) * (y - y_arr.at(i)) + S.at(i).at(2) * pow((y - y_arr.at(i)), 2)
                                                                                    + S.at(i).at(3) * pow((y - y_arr.at(i)), 3);           
                        circle(spline_image, Point(cvRound(x), y), 1, Scalar(255), -1);
                        points_to_cut[idx].push_back(Point(cvRound(x), y));
                    }

                }
            }    
        }

    
        for (int i = 0; i < x_arr.size(); i++)
        {
            // cout<<"x_arr[i].size() "<<x_arr[i].size()<<endl;
            // for (int j = 0; j < x_arr[i].size(); j++)
            // {
                cv::circle(spline_image, Point(x_arr[i], y_arr[i]), 1, Scalar(231, 191, 200), -1);
                // putText(spline_image, to_string(j), Point(x_arr[i].at(j), y_arr[i].at(j)), 1, 1, Scalar(255), 1);
                // cout<<Point(x_arr[i].at(j), y_arr[i].at(j))<<endl;
            // }
            // cout<<endl;
        }
}    
    imwrite("result_images/spline.png", spline_image);


    // vector<int> x1_arr, y1_arr;
    // int step1 = cvRound(sqrt(sides_add_points[id_cur_2].size()));
    // for (int r = 0; r < sides_points[id_cur_2].size(); r++)
    // {
    //     x1_arr.push_back(sides_points[id_cur_2][r].x);
    //     y1_arr.push_back(sides_points[id_cur_2][r].y);
    //     circle(side_points, sides_points[id_cur_2][r], 1, Scalar(255, 0, 0), -1);
    // }
    // circle(side_points, sides_points[id_cur_2][0], 1, Scalar(0, 0, 255), -1);
    // putText(side_points, to_string(0), sides_points[id_cur_2][0], 1, 1, Scalar(255), 1);
    // imwrite("result_images/side_points.png", side_points);
    
    // namedWindow("side points", WINDOW_NORMAL);
    // imshow("side points", side_points);
    // waitKey(0);
    // circle(lines_image, closest_points[id_cur_2+1], 7, Scalar(0,0,255), -1);
    // namedWindow("closest_points[id_cur_2+1]", WINDOW_NORMAL);
    // imshow("closest_points[id_cur_2+1]", lines_image);
    // waitKey(0);
    // if (norm(closest_points[id_cur_2+1] - Point(x1_arr.back(), y1_arr.back())) > 3)
    // {
    //     cout<<"krya\n";
    //     x1_arr.push_back(closest_points[id_cur_2+1].x);
    //     y1_arr.push_back(closest_points[id_cur_2+1].y);
    // }
    // for (int r = 0; r < sides_points[id_cur_2].size(); r++)
    // {
    //     x1_arr.push_back(sides_points[id_cur_2][r].x);
    //     y1_arr.push_back(sides_points[id_cur_2][r].y);
    // }
    // vector<vector<double>> S1;
    // vector<Point> right_points_to_cut;
    // int start1, end1;
    // cout<<"check\n";
    // cout<<x1_arr.size()<<" "<<y1_arr.size()<<endl;
    // cout<<x1_arr[0]<<" "<<x1_arr.back()<<endl;
    // cout<<y1_arr[0]<<" "<<y1_arr.back()<<endl;
    // cout<<abs(x1_arr[0] - x1_arr.back())<<endl;
    // cout<<abs(y1_arr[0] - y1_arr.back())<<endl;
    // if (abs(x1_arr[0] - x1_arr.back()) > abs(y1_arr[0] - y1_arr.back()))
    // {
    //     cout<<"horisontal1\n";
    //     S1 = spline(y1_arr, x1_arr);
    //     if(closest_points[id_cur_2].x > closest_points[(id_cur_2+1)%4].x)
    //     {
    //         start1 = (id_cur_2+1)%4;
    //         end1 = id_cur_2;
    //     }
    //     else
    //     {
    //         start1 = id_cur_2;
    //         end1 = (id_cur_2+1)%4;
    //     }
        
    //     // sort(x1_arr.begin(), x1_arr.end());
    //     // circle(spline_image, closest_points[start1], 15, Scalar(0,0,255), -1);
    //     for (int x = closest_points[start1].x; x < closest_points[end1].x+step; x++)
    //     {
    //         for (int i = 0; i < x1_arr.size()-1; i++)
    //         {
    //             if ((x >= x1_arr.at(i)) && (x < x1_arr.at(i+1)))
    //             {
    //                 double y = S1.at(i).at(0) + S1.at(i).at(1) * (x - x1_arr.at(i))
    //                                         + S1.at(i).at(2) * pow((x - x1_arr.at(i)), 2)
    //                                         + S1.at(i).at(3) * pow((x - x1_arr.at(i)), 3);
    //                 circle(spline_image, Point(cvRound(x), y), 2, Scalar(255), -1);
    //                 right_points_to_cut.push_back(Point(cvRound(x), y));
    //             }
    //         }
    //         // int i =  x1_arr.size() - 1;
    //         // if (x >= closest_points[end1].x)
    //         // {
    //         //     double y = S1.at(i-2).at(0) + S1.at(i-2).at(1) *     (x - x1_arr.at(i-1))
    //         //                                 + S1.at(i-2).at(2) * pow((x - x1_arr.at(i-1)), 2)
    //         //                                 + S1.at(i-2).at(3) * pow((x - x1_arr.at(i-1)), 3);

    //         //     // cout<<"y: "<<y<<" x: "<<x<<endl;
    //         //     circle(spline_image, Point(cvRound(x), y), 2, Scalar(255), -1);
    //         //     right_points_to_cut.push_back(Point(cvRound(x), y));

    //         // }            
    //     }    
    // }
    // else
    // {
    //     cout<<"vertical1\n";
    //     S1 = spline(x1_arr, y1_arr);
    //     // sort(y1_arr.begin(), y1_arr.end());
    //     if(closest_points[id_cur_2].y > closest_points[(id_cur_2+1)%4].y)
    //     {
    //         start1 = (id_cur_2+1)%4;
    //         end1 = id_cur_2;
    //     }
    //     else
    //     {
    //         start1 = id_cur_2;
    //         end1 = (id_cur_2+1)%4;
    //     }
    //     cout<<"start1 "<<start1<<" end1 "<<end1<<endl;

    //     // circle(spline_image, closest_points[start1], 5, Scalar(0,0,255), -1);
    //     // sort(y1_arr.begin(), y1_arr.end());

    //     for (int y = closest_points[start1].y; y < closest_points[end1].y+step; y++)
    //     {
    //         for (int i = 0; i < y1_arr.size()-1; i++)
    //         {
    //             if ((y >= y1_arr.at(i)) && (y < y1_arr.at(i+1)))
    //             {
    //                 double x = S1.at(i).at(0) + S1.at(i).at(1) * (y - y1_arr.at(i))
    //                                         + S1.at(i).at(2) * pow((y - y1_arr.at(i)), 2)
    //                                         + S1.at(i).at(3) * pow((y - y1_arr.at(i)), 3);

    //                 // cout<<"y: "<<y<<" x: "<<x<<endl;
    //                 circle(spline_image, Point(cvRound(x), y), 2, Scalar(255), -1);
    //                 right_points_to_cut.push_back(Point(cvRound(x), y));

    //             }
    //         }
    //         // int i = y1_arr.size()-1;
    //         // if (y > y1_arr.at(i))
    //         // {
    //         //     double x = S1.at(i-1).at(0) + S1.at(i-1).at(1) * (y - y1_arr.at(i))
    //         //                             + S1.at(i-1).at(2) * pow((y - y1_arr.at(i)), 2)
    //         //                             + S1.at(i-1).at(3) * pow((y - y1_arr.at(i)), 3);

    //         //     // cout<<"y: "<<y<<" x: "<<x<<endl;
    //         //     circle(spline_image, Point(cvRound(x), y), 2, Scalar(255), -1);
    //         //     right_points_to_cut.push_back(Point(cvRound(x), y));

    //         // }
    //     } 
    // }
    

    

    // for (int i = 0; i < x_arr.size(); i++)
    // {
    //     cv::circle(spline_image, Point(x_arr.at(i), y_arr.at(i)), 3, Scalar(231, 191, 200), -1);
    //     putText(spline_image, to_string(i), Point(x_arr.at(i), y_arr.at(i)), 1, 1, Scalar(255), 1);

    // }
    // for (int i = 0; i < x1_arr.size(); i++)
    // {
    //     cv::circle(spline_image, Point(x1_arr.at(i), y1_arr.at(i)), 3, Scalar(231, 191, 200), -1);
    //     // putText(spline_image, to_string(i), Point(x1_arr.at(i), y1_arr.at(i)) , 2,2, Scalar(0,0,255), 1);

    // }
    // imwrite("result_images/spline.png", spline_image);

    // namedWindow("spline", WINDOW_NORMAL);
    // imshow("spline", spline_image);
    // waitKey(0);

    cout<<"left point to cut "<<points_to_cut[0].size()<<endl;
    cout<<"right points to cut "<<points_to_cut[1].size()<<endl;

// // end аппроксимировать кривую

// // разделить на равномерные отрезки
if(points_to_cut[0].size() == 0 || points_to_cut[1].size() == 0)
{
    exit(0);
}
    float dist_to_arc, max_dist_to_arc = 0.0;
    cout<<"length "<<norm( points_to_cut[0].front() - points_to_cut[0].back())<<endl;


    for (int j = 0; j < points_to_cut.size(); j++)
    {
        dist_to_arc = 0.0;

        Point arc_begin = points_to_cut[j].front();
        Point arc_end = points_to_cut[j].back();

        
        for (int i = 0; i < points_to_cut[j].size(); i++)
        {

            Point arc_point = points_to_cut[j][i];

            float dist = abs(distancePt(arc_point, arc_begin, arc_end));
            dist_to_arc += dist;

        }
        cout<<"dist_to_arc "<<dist_to_arc<<endl;

        dist_to_arc /= points_to_cut[j].size();

        if (dist_to_arc > max_dist_to_arc)
        {
            max_dist_to_arc = dist_to_arc;
        }

    }
    

    cout<<"max_dist_to_arc "<<max_dist_to_arc<<endl;
    cout<<"max_dist_to_arc "<<ceil(max_dist_to_arc)<<endl;
    imwrite("result_images/arc_lines.png", arc_image);
// 
    int number_of_points = ceil(max_dist_to_arc);
    // int number_of_points = 7;
    cout<<"number_of_points "<<number_of_points<<endl;
// exit(0);
    vector<vector<Point>> test_vector(2);
    for (int i = 0; i < points_to_cut.size(); i++)
    {

        int size = points_to_cut[i].size();

        float step = static_cast<float>(size) / number_of_points;
        cout<<"step "<<step<<endl;

        for (int j = 0; j < number_of_points; j++) // подобрать общие делители
        {
            float val = j * step;
            int idx = cvRound(val) >= size ? size - 1 : cvRound(val);
            cout << j << " " << idx << " " << test_vector.size() << endl;
            test_vector[i].push_back(points_to_cut[i][idx]);
        }
        test_vector[i].push_back(points_to_cut[i].back());

    }


    // int left_step, right_step;
    // bool flag = false;
    // for (int i = 9; i < 40; i++)
    // {
    //     double temp_left_size = ceil(static_cast<double>(left_size) / i);
    //     for (int j = 9; j < 40; j++)
    //     {
    //         double temp_right_size = ceil(static_cast<double>(right_size) / j) ;
    //         // cout<<"temp left size "<<static_cast<double>(left_size)/i<<endl;
            
    //         // cout<<"temp right size "<<static_cast<double>(right_size)/j<<endl;
            
    //         if (temp_left_size == temp_right_size)
    //         {
    //             left_step = i;
    //             right_step = j;
    //             flag = true;
    //             break;
    //         }
    //     }
    //     if(flag) break;
    // }
    // cout<<"left step "<<left_step<<" right step "<<right_step<<endl;



    // vector<Point> right_test_vector, left_test_vector;

    // Mat test_mask = Mat::zeros(bin_barcode.rows, bin_barcode.cols, CV_8UC1);
    
    // for (float i = 0; i <= left_size; i+=left_step) // подобрать общие делители
    // {
    //     int idx = cvRound(i) >= left_size ? left_size - 1 : cvRound(i);
    //     cout << i << " " << idx << " " << left_test_vector.size() << endl;
    //     left_test_vector.push_back(left_points_to_cut[idx]);
    // }
    // // left_test_vector.push_back(left_points_to_cut.back());
    // cout<<"left test vector size = "<<left_test_vector.size()<<endl;
    // for (float i = 0; i <= right_size; i+=right_step)
    // {
    //     int idx = cvRound(i) >= right_size ? right_size - 1 : cvRound(i);
    //     cout << i << " " << idx << " " << right_test_vector.size() << endl;
    //     right_test_vector.push_back(right_points_to_cut[idx]);
    // }
    // for (int i = 0; i < number_of_points; i++) // подобрать общие делители
    // {
    //     float val = i * left_step;
    //     int idx = cvRound(val) >= left_size ? left_size - 1 : cvRound(val);
    //     cout << i << " " << idx << " " << left_test_vector.size() << endl;
    //     left_test_vector.push_back(left_points_to_cut[idx]);
    // }
    // left_test_vector.push_back(left_points_to_cut.back());
    cout<<"left test vector size = "<<test_vector[0].size()<<endl;
    // for (int i = 0; i < number_of_points; i++)
    // {
    //     float val = i * right_step;
    //     int idx = cvRound(val) >= right_size ? right_size - 1 : cvRound(val);
    //     cout << i << " " << idx << " " << right_test_vector.size() << endl;
    //     right_test_vector.push_back(right_points_to_cut[idx]);
    // }
    // right_test_vector.push_back(right_points_to_cut.back());
    cout<<"right test vector size = "<<test_vector[1].size()<<endl;

    // cout<<"left\n";
    // for(int i = 0; i < left_test_vector.size(); i++)
    // {
    //     circle(test_mask, left_test_vector[i], 3, Scalar(255), -1);
    //     cout<<i<<" "<<left_test_vector[i]<<endl;
    // }
    // cout<<"right\n";
    // for(int i = 0; i < right_test_vector.size(); i++)
    // {
    //     circle(test_mask, right_test_vector[i], 3, Scalar(255), -1);
    //     cout<<i<<" "<<right_test_vector[i]<<endl;
    // }
    // namedWindow("test", WINDOW_NORMAL);
    // imshow("test", test_mask);
    // waitKey(0);
    // if(left_points_to_cut.back() != left_test_vector.back()) left_test_vector.push_back(left_points_to_cut.back());
    // if(right_points_to_cut.back() != right_test_vector.back()) right_test_vector.push_back(right_points_to_cut.back());
    vector<Point> temp_right_convexhull;
    vector<Point> temp_left_convexhull;
    for (int i = 0; i < test_vector.size(); i++)
    {
        Point temp_point_start = test_vector[i].front();
        Point temp_point_end = test_vector[i].back();
        if (abs(temp_point_start.x - temp_point_end.x) > abs(temp_point_start.y - temp_point_end.y))
        {
            if(test_vector[i].front().y > test_vector[(i+1)%2].front().y)
            {
                cout<<"krya1 "<<i<<endl;
                temp_left_convexhull = test_vector[i];
                temp_right_convexhull = test_vector[(i+1)%2];
            }
            
        }
        else
        {

            if(test_vector[i].front().x < test_vector[(i+1)%2].front().x)
            {
                cout<<"krya2 "<<i<<endl;
                temp_left_convexhull = test_vector[i];
                temp_right_convexhull = test_vector[(i+1)%2];
            }
            

        }
        
    }
    
// // end разделить на равномерные отрезки


cout<<"temp_right_convexhull size "<<temp_right_convexhull.size()<<endl;
cout<<"temp_left_convexhull size "<<temp_left_convexhull.size()<<endl;
Point temp_point_start = temp_left_convexhull.front();
Point temp_point_end = temp_left_convexhull.back();
bool vertical_order;
if (abs(temp_point_start.x - temp_point_end.x) > abs(temp_point_start.y - temp_point_end.y))
{
    vertical_order = false;
}
else
{
    vertical_order = true;
}
cout<<"vertical_order "<<vertical_order<<endl;
    float test_perspective_size = 256.0;
    const Size temporary_size(cvRound(test_perspective_size), cvRound(test_perspective_size));
    Mat perspective_points_mask = Mat::zeros(temporary_size, CV_8UC3);
    Mat perspective_result = Mat::zeros(temporary_size, CV_8UC1);
    Mat temp_result = Mat(temporary_size, CV_8UC1, Scalar(255));
    vector<Point> original_points;
    cout<<"transformations\n";
    // namedWindow("test mask", WINDOW_NORMAL);
    // namedWindow("perspective test mask", WINDOW_NORMAL);
    // namedWindow("original points", WINDOW_NORMAL);
    // namedWindow("perspective_points_mask", WINDOW_NORMAL);
    // Mat result_perspective_mask = Mat::zeros(bin_barcode.rows, bin_barcode.cols, CV_8UC1);
    float start_cut = 0.0;
    float dist = test_perspective_size / (temp_right_convexhull.size() - 1);
    cout<<"dist "<<dist<<endl;

    for (int i = 1; i < temp_right_convexhull.size(); i++)
    {
        Mat test_mask = Mat::zeros(bin_barcode.rows, bin_barcode.cols, CV_8UC1);

        Point start_point = temp_left_convexhull[i];
        // circle(transform_image, start_point, 15, Scalar(255), -1);
        // putText(test_mask, "start point", start_point, 1, 1, Scalar(255), 1);
        Point prev_start_point = temp_left_convexhull[i-1];
        // circle(test_mask, prev_start_point, 2, Scalar(255), -1);
        // putText(test_mask, "prev start point", prev_start_point, 1, 1, Scalar(255), -1);
        Point finish_point = temp_right_convexhull[i];
        // circle(test_mask, finish_point, 2, Scalar(255), -1);
        Point prev_finish_point = temp_right_convexhull[i-1];
        // circle(test_mask, prev_finish_point, 2, Scalar(255), -1);
        Point center_1 = getMidSection(prev_start_point, prev_finish_point);
        Point center_2 = getMidSection(start_point, finish_point);
        line(transform_image, prev_start_point, finish_point, Scalar(216, 216, 86), 1);
        line(transform_image, prev_finish_point, start_point, Scalar(216, 216, 86), 1);
        line(image, start_point, finish_point, Scalar(216, 216, 86), 2);
        for (int j = 0; j < result_locations.size(); j++)
        {
            if ((pointPosition(start_point, finish_point, result_locations[j]) > 0) && 
                (pointPosition(prev_start_point, prev_finish_point, result_locations[j]) < 0))
            {
                // cout<<"krya\n";
                circle(test_mask, result_locations[j], 1, Scalar(255), -1);
            }
        }
        // if (i == temp_right_convexhull.size() - 1)
        // {
        //     for (int j = 0; j < result_locations.size(); j++)
        //     {
        //         if (pointPosition(prev_finish_point, prev_start_point, result_locations[j]) < 0)
        //         {
        //             // cout<<"krya\n";
        //             circle(test_mask, result_locations[j], 1, Scalar(255), -1);
        //         }
        //     }
        // }
        // imshow("test mask", test_mask);
        // waitKey(0);

        vector<Point2f> perspective_points;
        // cout<<"vertical_order "<<vertical_order<<endl;
        // if (vertical_order)
        // {
            perspective_points.push_back(Point2f(0.0, start_cut));
            perspective_points.push_back(Point2f(test_perspective_size, start_cut));

            perspective_points.push_back(Point2f(test_perspective_size, start_cut+dist));
            perspective_points.push_back(Point2f(0.0, start_cut+dist));

            perspective_points.push_back(Point2f(test_perspective_size * 0.5, start_cut + dist * 0.5));

        // }
        // else
        // {
        //     perspective_points.push_back(Point2f(0.0, start_cut+dist));
        //     perspective_points.push_back(Point2f(0.0, start_cut));

        //     perspective_points.push_back(Point2f(test_perspective_size, start_cut));
        //     perspective_points.push_back(Point2f(test_perspective_size, start_cut+dist));

        //     perspective_points.push_back(Point2f(test_perspective_size * 0.5, start_cut + dist * 0.5));

        // }
        
        for (int k = 0; k < perspective_points.size(); k++)
        {
            circle(perspective_points_mask, perspective_points[k], 3, Scalar(150, 0, 150), -1);
        }
        imwrite("result_images/perspective_points_mask.png", perspective_points_mask);

        // imshow("perspective_points_mask", perspective_points_mask);
        // waitKey(0);
        start_cut += dist;

        vector<Point> original_points;


        original_points.push_back(prev_start_point);
        original_points.push_back(prev_finish_point);
        original_points.push_back(finish_point);
        original_points.push_back(start_point);


        // Point center_point = getMidSection(center_1, center_2);
        Point center_point = intersectionLines(original_points[0], original_points[2],
                                            original_points[1], original_points[3]);
        // Point center_point = Point(center_point1.x, center_point2.y);
        vector<Point> pts = original_points;
        pts.push_back(center_point);
        for(int k = 0; k < pts.size(); k++)
        {
            circle(transform_image, pts[k], 3, Scalar(150, 200, 0), -1);
        }

        // imshow("original points", transform_image);
        // waitKey(0);

        Mat H = findHomography(pts, perspective_points);
        Mat intermediate, temp_intermediate;
        warpPerspective(test_mask, temp_intermediate, H, temporary_size, INTER_NEAREST);
        perspective_result += temp_intermediate;

        // imshow("perspective test mask", temp_intermediate);
        // waitKey(0);
    }
    imwrite("result_images/original_points.png", transform_image);
    imwrite("result_images/perspective_result.png", perspective_result);

    // namedWindow("perspective result", WINDOW_NORMAL);
    // imshow("perspective result", perspective_result);
    // waitKey(0);

    Mat finish_result = temp_result - perspective_result;
    Mat no_border_intermediate = finish_result(Range(1, finish_result.rows), Range(1, finish_result.cols));
    Mat intermediate;
    const int border = cvRound(0.1 * test_perspective_size);
    const int borderType = BORDER_CONSTANT;
    copyMakeBorder(no_border_intermediate, intermediate, border, border, border, border, borderType, Scalar(255));
    // namedWindow("finish_result", WINDOW_NORMAL);
    // imshow("finish_result", intermediate);
    // waitKey(0);
    cout<<"argv "<<argv[1]<<endl;
    String name = "result_images/finish_result_" + String(argv[1]);
    imwrite(name, intermediate);
    cout<<"write to "<<name<<endl;


    // namedWindow("lines to cut", WINDOW_NORMAL);
    // imshow("lines to cut", image);
    // waitKey(0);



  return result_angle_list;
}



inline Point computeOffset(const vector<Point>& v)
{
    // compute the width/height of convex hull
    Rect areaBox = boundingRect(v);

    // compute the good offset
    // the box is consisted by 7 steps
    // to pick the middle of the stripe, it needs to be 1/14 of the size
    const int cStep = 7 * 2;
    Point offset = Point(areaBox.width, areaBox.height);
    offset /= cStep;
    return offset;
}

int main( int argc, char** argv ){

  Mat image, barcode, bin_barcode, gray;
  image = imread(argv[1]);
  Mat resized_image;
  vector<Point2f> localization_points; //{Point(1207.8281, 2055.375), Point(1760.0625, 2034.7031), Point(1269.8438, 2542.6406)};
  vector<Point2f> transformation_points;
  double coeff_expansion;
  cout<<image.size()<<endl;
  const double min_side = std::min(image.size().width, image.size().height);
  if (min_side > 512.0)
  {
      coeff_expansion = min_side / 512.0;
      const int width  = cvRound(image.size().width  / coeff_expansion);
      const int height = cvRound(image.size().height  / coeff_expansion);
      Size new_size(width, height);
      resize(image, resized_image, new_size, 0, 0, INTER_LINEAR);
  }

  else
  {
      coeff_expansion = 1.0;
      resized_image = image.clone();
  }
  int incn = image.channels();
  if( incn == 3 || incn == 4 )
  {
      cvtColor(resized_image, gray, COLOR_BGR2GRAY);
  }
  adaptiveThreshold(gray, bin_barcode, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 83, 2);
  cout<<bin_barcode.size()<<endl;
//   namedWindow( "bin_barcode", WINDOW_NORMAL );
//   imshow( "bin_barcode", bin_barcode );
//   waitKey(0);
  //
cout<<resized_image.size()<<endl;


  double eps_vertical = 0.2;
  double eps_horizontal = 0.1;

  vector<Vec3d> result;
  const int height_bin_barcode = bin_barcode.rows;
  const int width_bin_barcode  = bin_barcode.cols;
  const size_t test_lines_size = 5;
  double test_lines[test_lines_size];
  vector<size_t> pixels_position;

  for (int y = 0; y < height_bin_barcode; y++)
  {
      pixels_position.clear();
      const uint8_t *bin_barcode_row = bin_barcode.ptr<uint8_t>(y);

      int pos = 0;
      for (; pos < width_bin_barcode; pos++) { if (bin_barcode_row[pos] == 0) break; }
      if (pos == width_bin_barcode) { continue; }

      pixels_position.push_back(pos);
      pixels_position.push_back(pos);
      pixels_position.push_back(pos);
      // for (int i = 0; i < pixels_position.size(); i++)
      // {
      //   cout<<y<<' '<<"pixels_position "<<pixels_position[i]<<endl;
      // }
      uint8_t future_pixel = 255;
      for (int x = pos; x < width_bin_barcode; x++)
      {
          if (bin_barcode_row[x] == future_pixel)
          {
              future_pixel = 255 - future_pixel;
              pixels_position.push_back(x);
          }
      }
      // for (int i = 0; i < pixels_position.size(); i++)
      // {
      //   cout<<y<<' '<<"pixels_position "<<pixels_position[i]<<endl;
      // }
      pixels_position.push_back(width_bin_barcode - 1);
      for (size_t i = 2; i < pixels_position.size() - 4; i+=2)
      {
          test_lines[0] = static_cast<double>(pixels_position[i - 1] - pixels_position[i - 2]);
          test_lines[1] = static_cast<double>(pixels_position[i    ] - pixels_position[i - 1]);
          test_lines[2] = static_cast<double>(pixels_position[i + 1] - pixels_position[i    ]);
          test_lines[3] = static_cast<double>(pixels_position[i + 2] - pixels_position[i + 1]);
          test_lines[4] = static_cast<double>(pixels_position[i + 3] - pixels_position[i + 2]);

          double length = 0.0, weight = 0.0;

          for (size_t j = 0; j < test_lines_size; j++) { length += test_lines[j]; }

          if (length == 0) { continue; }
          for (size_t j = 0; j < test_lines_size; j++)
          {
              if (j != 2) { weight += fabs((test_lines[j] / length) - 1.0/7.0); }
              else        { weight += fabs((test_lines[j] / length) - 3.0/7.0); }
          }

          if (weight < eps_vertical)
          {
              Vec3d line;
              line[0] = static_cast<double>(pixels_position[i - 2]);
              line[1] = y;
              line[2] = length;
              result.push_back(line);
          }
      }
  }

  cout<<result[0]<<endl;
//   for(size_t i = 0; i < result.size()-1; i++)
//   {
//     line(resized_image, Point2i(result.at(i)[0], result.at(i)[1]), Point2i(result.at(i)[0] + result.at(i)[3], result.at(i)[1]), Scalar(140,140,0), 3);
//   }
//   namedWindow( "lines", WINDOW_NORMAL );
//   imshow( "lines", resized_image );
//   waitKey(0);

  Point2f begin, end;
  //vector<Vec3d> list_lines_x = searchHorizontalLines();
  if( result.empty() ) { std::cout << "false result" << '\n'; return false; }
  vector<Point2f> list_lines_y = separateVerticalLines(resized_image, bin_barcode, result);
  if( list_lines_y.size() < 3 ) { std::cout << "false list_lines_y" << '\n';return false; }

  vector<Point2f> centers;
  Mat labels;
  kmeans(list_lines_y, 3, labels,
         TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1),
         3, KMEANS_PP_CENTERS, localization_points);

  fixationPoints(bin_barcode, localization_points);
  if (localization_points.size() != 3) { std::cout << "false localization_points" << '\n';return false; }

  if (coeff_expansion > 1.0)
  {
      const int width  = cvRound(bin_barcode.size().width  * coeff_expansion);
      const int height = cvRound(bin_barcode.size().height * coeff_expansion);
      Size new_size(width, height);
      Mat intermediate;
      resize(bin_barcode, intermediate, new_size, 0, 0, INTER_LINEAR);
      resize(resized_image, resized_image, new_size, 0, 0, INTER_LINEAR);
      bin_barcode = intermediate.clone();
      for (size_t i = 0; i < localization_points.size(); i++)
      {
          localization_points[i] *= coeff_expansion;
      }
  }

  for (size_t i = 0; i < localization_points.size(); i++)
  {
      for (size_t j = i + 1; j < localization_points.size(); j++)
      {
          if (norm(localization_points[i] - localization_points[j]) < 10)
          {
              return false;
          }
      }
  }

//   std::cout<<"localization_points\n"<<localization_points<<std::endl;
  for(size_t i = 0; i < localization_points.size(); i++)
  {        
    putText(resized_image, to_string(i), localization_points[i], 1, 2, Scalar(155, 20, 155), 1);

    circle(resized_image, Point2i(localization_points.at(i)), 2, Scalar(0,0,255), -1);
  }
//   namedWindow( "localization_points", WINDOW_NORMAL );
//   imshow( "intersection_points", resized_image );
//   waitKey(0);

  Mat mask, mask_roi;
  vector<Point> locations, non_zero_elem[3], newHull;
  vector<Point2f> new_non_zero_elem[3];
  vector<int> left_x_positions[3], right_x_positions[3], up_y_positions[3], down_y_positions[3];
  bool x_dist, y_dist;
//   namedWindow( "non_zero_elem", WINDOW_NORMAL );


  for (size_t i = 0; i < 3; i++)
  {
    // int test_white_length;

    // uint8_t future_pixel = 255;
    // int x_position = cvRound(localization_points[i].x);
    // int y_position = cvRound(localization_points[i].y);
    //
    // for (int y = y_position; y > 0; y--)
    // {
    //   const uint8_t *bin_barcode_row = bin_barcode.ptr<uint8_t>(y);
    //   if (bin_barcode_row[x_position] == future_pixel)
    //   {
    //     up_y_positions[i].push_back(y);
    //     future_pixel = 255 - future_pixel;
    //   }
    //   if (up_y_positions[i].size() == 3) { break; }
    // }
    //
    // future_pixel = 255;
    //
    // for (int y = y_position; y < bin_barcode.rows; y++)
    // {
    //   const uint8_t *bin_barcode_row = bin_barcode.ptr<uint8_t>(y);
    //   if (bin_barcode_row[x_position] == future_pixel)
    //   {
    //     down_y_positions[i].push_back(y);
    //     future_pixel = 255 - future_pixel;
    //   }
    //   if (down_y_positions[i].size() == 3) { break; }
    // }
    //
    // future_pixel = 255;
    //
    // for (int x = x_position; x > 0; x--)
    // {
    //   const uint8_t *bin_barcode_row = bin_barcode.ptr<uint8_t>(y_position);
    //   if (bin_barcode_row[x] == future_pixel)
    //   {
    //     left_x_positions[i].push_back(x);
    //     future_pixel = 255 - future_pixel;
    //   }
    //   if (left_x_positions[i].size() == 3) { break; }
    // }
    //
    // future_pixel = 255;
    //
    // for (int x = x_position; x < bin_barcode.cols; x++)
    // {
    //   const uint8_t *bin_barcode_row = bin_barcode.ptr<uint8_t>(y_position);
    //   if (bin_barcode_row[x] == future_pixel)
    //   {
    //     right_x_positions[i].push_back(x);
    //     future_pixel = 255 - future_pixel;
    //   }
    //   if (right_x_positions[i].size() == 3) { break; }
    // }

      mask = Mat::zeros(bin_barcode.rows + 2, bin_barcode.cols + 2, CV_8UC1);
      uint8_t next_pixel, future_pixel = 255;
      int count_test_lines = 0, index = cvRound(localization_points[i].x);
      for (; index < bin_barcode.cols - 1; index++)
      {
          next_pixel = bin_barcode.ptr<uint8_t>(cvRound(localization_points[i].y))[index + 1];
          if (next_pixel == future_pixel)
          {
              // x_positions[i].push_back((index+1));
              future_pixel = 255 - future_pixel;
              count_test_lines++;
              if (count_test_lines == 2)
              {
                  // index_black = index + 1;
                  // test_white_length = abs(index_black - index_white);
                  floodFill(bin_barcode, mask,
                            Point(index + 1, cvRound(localization_points[i].y)), 255,
                            0, Scalar(), Scalar(), FLOODFILL_MASK_ONLY);
                  break;
              }
          }
      }
      // cout<<"test_white_length "<<test_white_length<<endl;


      mask_roi = mask(Range(1, bin_barcode.rows - 1), Range(1, bin_barcode.cols - 1));
      findNonZero(mask_roi, non_zero_elem[i]);

      // cout<<"non zero elem i "<<i<<endl;
      // for (int j = 0; j < non_zero_elem[i].size(); j++) { cout<<non_zero_elem[i].at(j)<<endl;}

      // double left_x_check_dist = abs(cvRound(localization_points[i].x) - left_x_positions[i].back());
      // double right_x_check_dist = abs(cvRound(localization_points[i].x) - right_x_positions[i].back());
      // double up_y_check_dist = abs(cvRound(localization_points[i].y) - up_y_positions[i].back());
      // double down_y_check_dist = abs(cvRound(localization_points[i].y) - down_y_positions[i].back());
      vector<Point> indices;

      for (size_t j = 0; j < non_zero_elem[i].size(); j++)
      {
        // double elem_dist = norm(Point2i(cvRound(localization_points[i].x), cvRound(localization_points[i].y)) - non_zero_elem[i].at(j));
        // // cout<<"elem_dist "<<elem_dist<<endl;
        // if (elem_dist < test_white_length * 3.5)
        // {
          circle(mask_roi, non_zero_elem[i].at(j), 1, Scalar(255,255,255));
        //
        // // }
        // Mat mask_lines = Mat::zeros(bin_barcode.rows, bin_barcode.cols, CV_8UC1);
        // line(mask_lines, localization_points[i], non_zero_elem[i].at(j), Scalar(255,255,255), 1);
        //
        // vector<Point> line_pixels;
        // findNonZero(mask_lines, line_pixels);

        // for (int k = 0; k < line_pixels.size()-1; k++)
        // {
        //   line(resized_image, localization_points[i], line_pixels.at(k), Scalar(50+j,0,0), 1);
        // }
        // imshow( "line_pixels", resized_image );
        // waitKey(0);
        //
        // Point white_pixel, white_back_pixel, black_pixel, black_back_pixel;
        // uint8_t fpixel = 255;
        // int count_pixels = 0;
        // for (int k = 1; k < line_pixels.size(); k++)
        // {
        //     if (bin_barcode.ptr<uint8_t>(line_pixels.at(k).y)[line_pixels.at(k).x] == fpixel)
        //     {
        //         fpixel = 255 - fpixel;
        //         count_pixels++;
        //     }
        //     if (count_pixels == 1) { white_pixel = Point(line_pixels.at(k)); white_back_pixel = Point(line_pixels.at(k-1));  }
        //     if (count_pixels == 2) { black_pixel = Point(line_pixels.at(k)); black_back_pixel = Point(line_pixels.at(k-1)); break;}
        // }
        // circle(resized_image, white_pixel, 1, Scalar(255,0,255));
        // circle(resized_image, black_pixel, 1, Scalar(255,255,0));
        // imshow( "pixels", resized_image );
        // waitKey(0);
        // double test_white_length = norm(white_pixel - black_back_pixel);
        // double elem_dist;
        // if ((non_zero_elem[i].at(j).y < localization_points[i].y) || ((non_zero_elem[i].at(j).y = localization_points[i].y) && (non_zero_elem[i].at(j).x < localization_points[i].x)) )
        // {
        //   elem_dist = norm(white_back_pixel - non_zero_elem[i].at(j));
        // }
        // else if ((non_zero_elem[i].at(j).y > localization_points[i].y ) || ((non_zero_elem[i].at(j).y = localization_points[i].y) && (non_zero_elem[i].at(j).x > localization_points[i].x)))
        // {
        //   elem_dist = norm(black_pixel - non_zero_elem[i].at(j));
        // }
        // cout<<"test_white_length "<<test_white_length<<endl;
        // cout<<"elem_dist "<<elem_dist<<endl;
        //
        // if (elem_dist * 0.3 <= test_white_length )
        // {
        //   circle(mask_roi, non_zero_elem[i].at(j), 1, Scalar(255,255,255));
        //   indices.push_back(non_zero_elem[i].at(j));
        // }

      //   x_dist = false, y_dist = false;
      //   double x_elem_dist = (localization_points[i].x - non_zero_elem[i].at(j).x);
      //   double y_elem_dist = (localization_points[i].y - non_zero_elem[i].at(j).y);
      //   if (x_elem_dist > 0)
      //   {
      //     if (abs(x_elem_dist) <= left_x_check_dist) { x_dist = true; }
      //   }
      //   else
      //   {
      //     if (abs(x_elem_dist) <= right_x_check_dist) { x_dist = true; }
      //   }
      //   if (y_elem_dist > 0)
      //   {
      //     if (abs(y_elem_dist) <= down_y_check_dist) { y_dist = true; }
      //   }
      //   else
      //   {
      //     if (abs(y_elem_dist) <= up_y_check_dist) {y_dist = true;}
      //   }
      //
      //   if ((x_dist == true) && (y_dist == true))
      //   {
      //     indices.push_back(non_zero_elem[i].at(j));
      //     // circle(mask_roi, non_zero_elem[i].at(j), 2, Scalar(255,255,255));
      //   }
      }
      // non_zero_elem[i] = indices;

      // for (int j=0; j < non_zero_elem[i].size(); j++)
      // {
        // circle(mask_roi, non_zero_elem[i].at(j), 1, Scalar(255,255,255));
      // }
    //   imshow( "non_zero_elem", mask_roi );
    //   waitKey(0);

      newHull.insert(newHull.end(), non_zero_elem[i].begin(), non_zero_elem[i].end());

  }

  convexHull(newHull, locations);
  for (size_t i = 0; i < locations.size(); i++)
  {
      for (size_t j = 0; j < 3; j++)
      {
          for (size_t k = 0; k < non_zero_elem[j].size(); k++)
          {
              if (locations[i] == non_zero_elem[j][k])
              {
                  new_non_zero_elem[j].push_back(locations[i]);
              }
          }
      }
  }







  // Mat drawing = Mat::zeros(bin_barcode.size(), CV_8UC1);
  // vector<vector<Point>> tmp;
  // tmp.push_back(locations);
  // drawContours(drawing, tmp, 0, (255, 0, 0));
//   cout<<"new_non_zero_elem\n"<<new_non_zero_elem<<endl;
//   cout<<"hull_locations\n"<<locations<<endl;
  cout<<locations.size()<<endl;
//   for (int j=0; j < locations.size() - 1; j++)
//   {
//     circle(resized_image, locations.at(j), 5, Scalar(0,255,0));
//   }
//   namedWindow( "locations", WINDOW_NORMAL );
//   imshow( "locations", resized_image );
//   waitKey(0);
//   for (int j=0; j < newHull.size(); j++)
//   {
//     circle(resized_image, newHull.at(j), 2, Scalar(0,255,0), 3);
//   }
//   namedWindow( "newHull", WINDOW_NORMAL );
//   imshow( "newHull", resized_image );
//   waitKey(0);


  double pentagon_diag_norm = -1;
  Point2f down_left_edge_point, up_right_edge_point, up_left_edge_point;
  for (size_t i = 0; i < new_non_zero_elem[1].size(); i++)
  {
      for (size_t j = 0; j < new_non_zero_elem[2].size(); j++)
      {
          double temp_norm = norm(new_non_zero_elem[1][i] - new_non_zero_elem[2][j]);
          if (temp_norm > pentagon_diag_norm)
          {
              down_left_edge_point = new_non_zero_elem[1][i];
              up_right_edge_point  = new_non_zero_elem[2][j];
              pentagon_diag_norm = temp_norm;
          }
      }
  }
//   cout<<down_left_edge_point<<endl;
//   cout<<up_right_edge_point<<endl;
//   cout<<up_left_edge_point<<endl;



  if (down_left_edge_point == Point2f(0, 0) ||
      up_right_edge_point  == Point2f(0, 0) ||
      new_non_zero_elem[0].size() == 0) { return false; }

  double max_area = -1;
  up_left_edge_point = new_non_zero_elem[0][0];

  for (size_t i = 0; i < new_non_zero_elem[0].size(); i++)
  {
      vector<Point2f> list_edge_points;
      list_edge_points.push_back(new_non_zero_elem[0][i]);
      list_edge_points.push_back(down_left_edge_point);
      list_edge_points.push_back(up_right_edge_point);

      double temp_area = fabs(contourArea(list_edge_points));
      if (max_area < temp_area)
      {
          up_left_edge_point = new_non_zero_elem[0][i];
          max_area = temp_area;
        //   cout<<"max_area"<<max_area<<endl;
      }
  }

//   circle(resized_image, down_left_edge_point, 5, Scalar(245,209,101), -1);
//   circle(resized_image, up_right_edge_point, 5, Scalar(245,209,101), -1);
//   circle(resized_image, up_left_edge_point, 5, Scalar(245,209,101), -1);
//   namedWindow( "edge_point", WINDOW_NORMAL );
//   imshow( "edge_point", resized_image );
//   waitKey(0);



  Point2f down_max_delta_point, up_max_delta_point;
  double norm_down_max_delta = -1, norm_up_max_delta = -1;
for (size_t i = 0; i < new_non_zero_elem[1].size(); i++)
{
    double temp_norm_delta = norm(up_left_edge_point - new_non_zero_elem[1][i])
                            + norm(down_left_edge_point - new_non_zero_elem[1][i]);
    if (norm_down_max_delta < temp_norm_delta)
    {
        down_max_delta_point = new_non_zero_elem[1][i];
        norm_down_max_delta = temp_norm_delta;
    }
}
for (size_t i = 0; i < new_non_zero_elem[2].size(); i++)
{
    double temp_norm_delta = norm(up_left_edge_point - new_non_zero_elem[2][i])
                            + norm(up_right_edge_point - new_non_zero_elem[2][i]);
    if (norm_up_max_delta < temp_norm_delta)
    {
        up_max_delta_point = new_non_zero_elem[2][i];
        norm_up_max_delta = temp_norm_delta;
    }
}

//   cout<<"down_max_delta_point"<<down_max_delta_point<<endl;
//   cout<<"up_max_delta_point"<<up_max_delta_point<<endl;
//   cout<<"norm_up_max_delta "<<norm_up_max_delta<<endl;
//   cout<<"norm_down_max_delta "<<norm_down_max_delta<<endl;  
//     cout<<abs(norm_down_max_delta-norm_up_max_delta)<<endl;
//   circle(resized_image, down_max_delta_point, 5, Scalar(35,247,252), -1);
//   circle(resized_image, up_max_delta_point, 5, Scalar(35,247,252), -1);
//   namedWindow( "max_delta_point", WINDOW_NORMAL );
//   imshow( "max_delta_point", resized_image );
//   waitKey(0);

  transformation_points.push_back(down_left_edge_point);
  transformation_points.push_back(up_left_edge_point);
  transformation_points.push_back(up_right_edge_point);
  transformation_points.push_back(
      intersectionLines(down_left_edge_point, down_max_delta_point,
                        up_right_edge_point, up_max_delta_point));

  // transformation_points[2] = pointProjection(transformation_points[2], transformation_points[3], transformation_points[1]);

  //
  // if (coeff_expansion > 1.0)
  // {
  //
  //     for (size_t i = 0; i < transformation_points.size(); i++)
  //     {
  //         transformation_points[i] *= coeff_expansion;
  //     }
  // }

  hull_mask = Mat::zeros(bin_barcode.rows, bin_barcode.cols, CV_8UC1);
  for (size_t i = 0; i < newHull.size(); i++)
  {
    circle(hull_mask, newHull.at(i), 1, Scalar(255));
  }

//   namedWindow( "hull_mask", WINDOW_NORMAL );
//   imshow( "hull_mask", hull_mask );
//   waitKey(0);

  
  
  vector<Point2f> white_pixels;
  findNonZero(hull_mask, white_pixels);
//   cout<<"white pixels "<<hull_mask.at(white_pixels[0])<<endl;

  vector<vector<Point> > hull_contours;
  findContours( hull_mask, hull_contours,  RETR_EXTERNAL , CHAIN_APPROX_NONE, Point(0, 0) );
  // vector< vector<Point> > hull(contours.size());


  Mat ones1 = Mat(bin_barcode.size(), CV_8UC1, 255);
  Mat ones2 = Mat(bin_barcode.size(), CV_8UC1, 255);

  ones1.at<unsigned char>(0, 0) = 0;
  ones2.at<unsigned char>(1, 0) = 0;

  Mat res = ones1 + ones2;

//   cout << (int)res.at<unsigned char>(0, 0) << " " << (int)res.at<unsigned char>(1,0) << " " << (int)res.at<unsigned char>(2,2) << endl;
vector<vector<Point>> temp_hull_contours;
cout<<"hull_contours size "<<hull_contours.size()<<endl;
for (int i = 0; i < localization_points.size(); i++)
{
    for (int j = 0; j < hull_contours.size(); j++)
    {
    
        if (pointPolygonTest(hull_contours[j], localization_points[i], false)>0)
        {
            temp_hull_contours.push_back(hull_contours[j]);
        }
        printf("hull_contours %d size %d\n", j, (int) hull_contours[j].size());
    }
}
hull_contours = temp_hull_contours;
  // for(int i = 0; i < hull_contours.size(); i++) convexHull(Mat(contours[i]), hull[i]);
  // create a blank image (black image)



  Mat hull_drawing = Mat::zeros(bin_barcode.size(), CV_8UC3);

//   for(int i = 0; i < hull_contours.size(); i++) {
//       Scalar color_contours = Scalar(0, 100 + i*50, 0); // green - color for contours
//       Scalar color = Scalar(255, 0, 0); // blue - color for convex hull
//       // draw ith contour
//       drawContours(hull_drawing, hull_contours, i, color_contours, 1, 8, vector<Vec4i>(), 0, Point());
//       // putText(drawing, to_string(i), contours[i][0], 1, 1, Scalar(143,143,143), 1);
//       // draw ith convex hull
//       // drawContours(hull_drawing, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point());
//   }
//   namedWindow( "hull_contours", WINDOW_NORMAL );
//   imshow( "hull_contours", hull_drawing );
//   waitKey(0);



  Mat original, no_border_intermediate, intermediate, straight;
  vector<Point2f> original_points;
  std::string result_info;
  uint8_t version = 0, version_size = 0;
  float test_perspective_size = 251;


  Mat inarr = image.clone();


// std::cout<<"krya\n";
  vector<Point2f> quadrilateral = getQuadrilateral(resized_image, bin_barcode, inarr, hull_contours, transformation_points, argv);
//   transformation_points = quadrilateral;

  Mat decode_gray;
  cvtColor(inarr, decode_gray, COLOR_BGR2GRAY);
  inarr = decode_gray;


  original = inarr.clone();

  original_points = transformation_points;

  // ПОСЧИТАТЬ УГЛЫ
  // double cos_quadrilateral[4], norm_quadrilateral[4];
  // Point2d coord_vectors[4];
  //
  // coord_vectors[0] = transformation_points[1] - transformation_points[0];
  // coord_vectors[1] = transformation_points[2] - transformation_points[1];
  // coord_vectors[2] = transformation_points[3] - transformation_points[2];
  // coord_vectors[3] = transformation_points[0] - transformation_points[3];
  //
  // norm_quadrilateral[0] = norm(coord_vectors[0]);
  // norm_quadrilateral[1] = norm(coord_vectors[1]);
  // norm_quadrilateral[2] = norm(coord_vectors[2]);
  // norm_quadrilateral[3] = norm(coord_vectors[3]);
  //
  // cos_quadrilateral[0] = (coord_vectors[0].x * (-1) * coord_vectors[3].x + coord_vectors[0].y * (-1) * coord_vectors[3].y) / (norm_triangl[2] * norm_quadrilateral[1]);
  // cos_quadrilateral[1] = cos_angles[1];
  // cos_quadrilateral[2] = (coord_vectors[1].x * (-1) * coord_vectors[2].x + coord_vectors[1].y * (-1) * coord_vectors[2].y) / (norm_triangl[0] * norm_quadrilateral[0]);
  // cos_quadrilateral[3] = (coord_vectors[2].x * (-1) * coord_vectors[3].x + coord_vectors[2].y * (-1) * coord_vectors[3].y) / (norm_quadrilateral[0] * norm_quadrilateral[1]);

  // cout<<"local_point\n";
  //
  // for (size_t i = 0; i < 4; i++)
  // {
  //   cout<<local_point[i]<<endl;
  // }
  // cout<<"cos_quadrilateral\n";
  // for (size_t i = 0; i < 4; i++)
  // {
  //   putText(bin_barcode, to_string(cos_quadrilateral[i]), local_point[i], 1, 1, Scalar(143,143,143), 2);
  //   cout<<cos_quadrilateral[i]<<endl;
  // }
  // namedWindow( "cos_quadrilateral", WINDOW_NORMAL );
  // imshow( "cos_quadrilateral", bin_barcode );
  // waitKey(0);


  //
//   for (size_t j=0; j < quadrilateral.size(); j++)
//   {
//     line(image, quadrilateral.at(j%quadrilateral.size()), quadrilateral.at((j+1)%quadrilateral.size()), Scalar(255,0,0), 3);
//   }
//   // line(image, transformation_points.at(3), transformation_points.at(0), Scalar(255,0,0), 6);
//   namedWindow( "transformation_points", WINDOW_NORMAL );
//   imshow( "transformation_points", image );
//   waitKey(0);










//   namedWindow( "original", WINDOW_NORMAL );
//   imshow( "original", original );
//   waitKey(0);

  const Point2f centerPt = intersectionLines(original_points[0], original_points[2],
                                                       original_points[1], original_points[3]);
  // if (cvIsNaN(centerPt.x) || cvIsNaN(centerPt.y))
  //     return false;

//   cout<<"center "<<centerPt<<endl;

  const Size temporary_size(cvRound(test_perspective_size), cvRound(test_perspective_size));

  vector<Point2f> perspective_points;
  perspective_points.push_back(Point2f(0.f, 0.f));
  perspective_points.push_back(Point2f(test_perspective_size, 0.f));

  perspective_points.push_back(Point2f(test_perspective_size, test_perspective_size));
  perspective_points.push_back(Point2f(0.f, test_perspective_size));

  perspective_points.push_back(Point2f(test_perspective_size * 0.5f, test_perspective_size * 0.5f));

//   for (size_t j=0; j < perspective_points.size()-1; j++)
//   {
//     line(resized_image, perspective_points.at(j), perspective_points.at(j+1), Scalar(153,0,153), 6);
//   }
//   line(resized_image, perspective_points.at(3), perspective_points.at(0), Scalar(153,0,153), 6);
//   namedWindow( "perspective_points", WINDOW_NORMAL );
//   imshow( "perspective_points", resized_image );
//   waitKey(0);

  vector<Point2f> pts = original_points;
  pts.push_back(centerPt);

  Mat H = findHomography(pts, perspective_points);
  Mat bin_original;
  adaptiveThreshold(original, bin_original, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 83, 2);
  Mat temp_intermediate;
  warpPerspective(bin_original, temp_intermediate, H, temporary_size, INTER_NEAREST);
  no_border_intermediate = temp_intermediate(Range(1, temp_intermediate.rows), Range(1, temp_intermediate.cols));

  const int border = cvRound(0.1 * test_perspective_size);
  const int borderType = BORDER_CONSTANT;
  copyMakeBorder(no_border_intermediate, intermediate, border, border, border, border, borderType, Scalar(255));


  LineIterator line_iter(intermediate, Point2f(0, 0), Point2f(test_perspective_size, test_perspective_size));
  Point black_point = Point(0, 0);
  for(int j = 0; j < line_iter.count; j++, ++line_iter)
  {
      const uint8_t value = intermediate.at<uint8_t>(line_iter.pos());
      if (value == 0) { black_point = line_iter.pos(); break; }
  }

  Mat decode_mask = Mat::zeros(intermediate.rows + 2, intermediate.cols + 2, CV_8UC1);
  floodFill(intermediate, decode_mask, black_point, 255, 0, Scalar(), Scalar(), FLOODFILL_MASK_ONLY);

  vector<Point> decode_locations, decode_non_zero_elem;
  Mat decode_mask_roi = decode_mask(Range(1, intermediate.rows - 1), Range(1, intermediate.cols - 1));
  findNonZero(decode_mask_roi, decode_non_zero_elem);
  convexHull(decode_non_zero_elem, decode_locations);
  Point offset = computeOffset(decode_locations);

  Point temp_remote = decode_locations[0], remote_point;
  const Point delta_diff = offset;
  for (size_t i = 0; i < decode_locations.size(); i++)
  {
      if (norm(black_point - temp_remote) <= norm(black_point - decode_locations[i]))
      {
          const uint8_t value = intermediate.at<uint8_t>(temp_remote - delta_diff);
          temp_remote = decode_locations[i];
          if (value == 0) { remote_point = temp_remote - delta_diff; }
          else { remote_point = temp_remote - (delta_diff / 2); }
      }
  }

  size_t transition_x = 0 , transition_y = 0;

  uint8_t future_pixel = 255;
  const uint8_t *intermediate_row = intermediate.ptr<uint8_t>(remote_point.y);
  for(int i = remote_point.x; i < intermediate.cols; i++)
  {
      if (intermediate_row[i] == future_pixel)
      {
          future_pixel = 255 - future_pixel;
          transition_x++;
      }
  }

  future_pixel = 255;
  for(int j = remote_point.y; j < intermediate.rows; j++)
  {
      const uint8_t value = intermediate.at<uint8_t>(Point(j, remote_point.x));
      if (value == future_pixel)
      {
          future_pixel = 255 - future_pixel;
          transition_y++;
      }
  }

  version = saturate_cast<uint8_t>((std::min(transition_x, transition_y) - 1) * 0.25 - 1);
  if ( !(  0 < version && version <= 40 ) ) { return false; }
  version_size = 21 + (version - 1) * 4;

//   cout<<"version "<<version<<endl;

  const double multiplyingFactor = (version < 3)  ? 1 :
                                   (version == 3) ? 1.5 :
                                   version * (5 + version - 4);
  const Size newFactorSize(
                cvRound(no_border_intermediate.size().width  * multiplyingFactor),
                cvRound(no_border_intermediate.size().height * multiplyingFactor));
  Mat postIntermediate(newFactorSize, CV_8UC1);
  resize(no_border_intermediate, postIntermediate, newFactorSize, 0, 0, INTER_AREA);

  const int delta_rows = cvRound((postIntermediate.rows * 1.0) / version_size);
  const int delta_cols = cvRound((postIntermediate.cols * 1.0) / version_size);

  vector<double> listFrequencyElem;
  for (int r = 0; r < postIntermediate.rows; r += delta_rows)
  {
      for (int c = 0; c < postIntermediate.cols; c += delta_cols)
      {
          Mat tile = postIntermediate(
                         Range(r, min(r + delta_rows, postIntermediate.rows)),
                         Range(c, min(c + delta_cols, postIntermediate.cols)));
          const double frequencyElem = (countNonZero(tile) * 1.0) / tile.total();
          listFrequencyElem.push_back(frequencyElem);
      }
  }

  double dispersionEFE = std::numeric_limits<double>::max();
  double experimentalFrequencyElem = 0;
  for (double expVal = 0; expVal < 1; expVal+=0.001)
  {
      double testDispersionEFE = 0.0;
      for (size_t i = 0; i < listFrequencyElem.size(); i++)
      {
          testDispersionEFE += (listFrequencyElem[i] - expVal) *
                               (listFrequencyElem[i] - expVal);
      }
      testDispersionEFE /= (listFrequencyElem.size() - 1);
      if (dispersionEFE > testDispersionEFE)
      {
          dispersionEFE = testDispersionEFE;
          experimentalFrequencyElem = expVal;
      }
  }

  straight = Mat(Size(version_size, version_size), CV_8UC1, Scalar(0));
  for (int r = 0; r < version_size * version_size; r++)
  {
      int i   = r / straight.cols;
      int j   = r % straight.cols;
      straight.ptr<uint8_t>(i)[j] = (listFrequencyElem[r] < experimentalFrequencyElem) ? 0 : 255;
  }


  // cout<<"straight"<<straight<<endl;
//   namedWindow( "straight", WINDOW_NORMAL );
//   imshow( "straight", straight );
//   waitKey(0);


  #ifdef HAVE_QUIRC
      if (straight.empty()) { return false; }

      quirc_code qr_code;
      memset(&qr_code, 0, sizeof(qr_code));

      qr_code.size = straight.size().width;
      for (int x = 0; x < qr_code.size; x++)
      {
          for (int y = 0; y < qr_code.size; y++)
          {
              int position = y * qr_code.size + x;
              qr_code.cell_bitmap[position >> 3]
                  |= straight.ptr<uint8_t>(y)[x] ? 0 : (1 << (position & 7));
          }
      }

      quirc_data qr_code_data;
      quirc_decode_error_t errorCode = quirc_decode(&qr_code, &qr_code_data);
      if (errorCode != 0) { return false; }

      for (int i = 0; i < qr_code_data.payload_len; i++)
      {
          result_info += qr_code_data.payload[i];
      }
      return true;
  // #else
      // return false;
  #endif

  Mat straight_qrcode;
  straight.convertTo(straight_qrcode,straight_qrcode.type());

//   namedWindow( "straight_qrcode", WINDOW_NORMAL );
//   imshow( "straight_qrcode", straight_qrcode );
//   waitKey(0);

  cout<<"result_info "<<result_info<<endl;

































  // Rect cont_rect = boundingRect(transformation_points);
  // cout<<"cont_rect "<<cont_rect.width<<cont_rect.height<<endl;
  // rectangle(image,cont_rect,Scalar(255,0,0),5,8,0);
  // namedWindow( "cont_rect", WINDOW_NORMAL );
  // imshow( "cont_rect", image );
  // waitKey(0);


  // Mat crop_gray, crop_bin_barcode;
  // Mat cropped_image = image(cont_rect);
  // namedWindow( "cropped_image", WINDOW_NORMAL );
  // imshow( "cropped_image", cropped_image );
  // waitKey(0);

  // int incn_crop = cropped_image.channels();
  // if( incn_crop == 3 || incn_crop == 4 )
  // {
  //
  //     cvtColor(cropped_image, crop_gray, COLOR_BGR2GRAY);
  // }
  // adaptiveThreshold(crop_gray, crop_bin_barcode, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 153, 6);
  //
  // vector<vector<Point> > code_contours;
  // vector<Vec4i> code_hierarchy;
  // находим контуры
    // findContours( crop_bin_barcode, code_contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE, Point(0,0));

  // create hull array for convex hull points
  // vector< vector<Point> > code_hull(code_contours.size());
  // for(int i = 0; i < code_contours.size(); i++) {
  //   cout<<"code_contours "<<code_contours[i]<<endl;
  //   convexHull(Mat(code_contours[i]), code_hull[i]);
  // }

  // create a blank image (black image)
  // Mat code_drawing = Mat::zeros(cropped_image.size(), CV_8UC3);

  // for(int i = 0; i < code_contours.size(); i++) {
  //     Scalar color_contours = Scalar(0, 255, 0); // green - color for contours
  //     Scalar color = Scalar(255, 0, 0); // blue - color for convex hull
  //     // draw ith contour
  //     drawContours(cropped_image, code_contours, i, color_contours, 1, 8, vector<Vec4i>(), 0, Point());
  //     // draw ith convex hull
  //     drawContours(cropped_image, code_hull, i, color, 1, 8, vector<Vec4i>(), 0, Point());
  // }
  // namedWindow( "crop_contours", WINDOW_NORMAL );
  // imshow( "crop_contours", cropped_image );
  // waitKey(0);
  //
  // vector<Point> code_merged_contour_points;
  // for (int i = 0; i < code_contours.size(); i++) {
  //       for (int j = 0; j < code_contours.at(i).size(); j++)
  //         code_merged_contour_points.push_back(code_contours.at(i).at(j));
  //
  // }
  //
  // vector< vector<Point> > code_hull(code_contours.size());
  // for(int i = 0; i < code_contours.size(); i++) {
  //   cout<<"code_contours "<<code_contours[i]<<endl;
  //   convexHull(Mat(code_merged_contour_points[i]), code_hull[i]);
  // }
  // for(int i = 0; i < code_hull.size(); i++) {
  //     Scalar color_contours = Scalar(0, 255, 0); // green - color for contours
  //     Scalar color = Scalar(255, 0, 0); // blue - color for convex hull
  //     // draw ith contour
  //     //drawContours(cropped_image, code_contours, i, color_contours, 1, 8, vector<Vec4i>(), 0, Point());
  //     // draw ith convex hull
  //     drawContours(cropped_image, code_hull, i, color, 1, 8, vector<Vec4i>(), 0, Point());
  // }
  // namedWindow( "crop_contours", WINDOW_NORMAL );
  // imshow( "crop_contours", cropped_image );
  // waitKey(0);
  //
  // // for(int i = 0; i < code_merged_contour_points.size(); i++) {
  // //     Scalar color_contours = Scalar(0, 0, 255); // green - color for contours
  // //     // draw ith contour
  // //     drawContours(cropped_image, code_merged_contour_points, i, color_contours, 1, 8, vector<Vec4i>(), 0, Point());
  // //
  // // }
  // namedWindow( "code_contours_poly", WINDOW_NORMAL );
  // imshow( "code_contours_poly", cropped_image );
  // waitKey(0);
  //vector<Point2f> quadrilateral = getQuadrilateral(transformation_points);
  //transformation_points = quadrilateral;
  //line(image, down_left_edge_point, up_right_edge_point, Scalar(0,255,0), 3);


  // vector<Point> locations, non_zero_elem[3], newHull;
  // vector<Point2f> new_non_zero_elem[3];
  // for (size_t i = 0; i < 3; i++)
  // {
  //     Mat mask = Mat::zeros(bin_barcode.rows + 2, bin_barcode.cols + 2, CV_8UC1);
  //     uint8_t next_pixel, future_pixel = 255;
  //     int count_test_lines = 0, index = cvRound(localization_points[i].x);
  //     for (; index < bin_barcode.cols - 1; index++)
  //     {
  //         next_pixel = bin_barcode.ptr<uint8_t>(cvRound(localization_points[i].y))[index + 1];
  //         if (next_pixel == future_pixel)
  //         {
  //             future_pixel = 255 - future_pixel;
  //             count_test_lines++;
  //             if (count_test_lines == 2)
  //             {
  //                 cout<<"aaaaaa"<<endl;
  //                 floodFill(bin_barcode, mask,
  //                           Point(index + 1, cvRound(localization_points[i].y)), 255,
  //                           0, Scalar(), Scalar(), FLOODFILL_MASK_ONLY);
  //
  //                 break;
  //             }
  //         }
  //     }
  //
  //     cout<<mask.size()<<endl;
  //     cout<<"mask"<<endl;
  //     namedWindow( "Display mask", WINDOW_NORMAL );
  //     imshow( "Display mask", mask );
  //     waitKey(0);
  //
  //     Mat mask_roi = mask(Range(1, bin_barcode.rows - 1), Range(1, bin_barcode.cols - 1));
  //     cout<<"mask_roi"<<endl;
  //     namedWindow( "Display mask_roi", WINDOW_NORMAL );
  //     imshow( "Display mask_roi", mask_roi );
  //     waitKey(0);
  //     findNonZero(mask_roi, non_zero_elem[i]);
  //     cout<<"non_zero_elem"<<non_zero_elem[i]<<endl;
  //     newHull.insert(newHull.end(), non_zero_elem[i].begin(), non_zero_elem[i].end());
  //     cout<<newHull<<endl;
  //
  // }
  // namedWindow( "Display window", WINDOW_NORMAL );
  // imshow( "Display window", bin_barcode );
  // waitKey(0);



  return 0;
}
