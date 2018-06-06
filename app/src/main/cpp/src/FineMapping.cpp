//
// Created by 庾金科 on 22/09/2017.
//

#include "FineMapping.h"
namespace pr{

    const int FINEMAPPING_H = 50;
    const int FINEMAPPING_W = 120;
    //const int PADDING_UP_DOWN = 30;

    const int PADDING_UP_DOWN = 0;

    void drawRect(cv::Mat image,cv::Rect rect) {
        cv::Point p1(rect.x,rect.y);
        cv::Point p2(rect.x+rect.width,rect.y+rect.height);
        cv::rectangle(image,p1,p2,cv::Scalar(0,255,0),1);
    }


    FineMapping::FineMapping(std::string prototxt,std::string caffemodel) {
         net = cv::dnn::readNetFromCaffe(prototxt, caffemodel);
    }

    cv::Mat FineMapping::FineMappingHorizon(cv::Mat FinedVertical,
                                            int leftPadding,
                                            int rightPadding) {

//        if(FinedVertical.channels()==1)
//            cv::cvtColor(FinedVertical,FinedVertical,cv::COLOR_GRAY2BGR);
        cv::Mat inputBlob = cv::dnn::blobFromImage(FinedVertical, 1/255.0, cv::Size(66,16),
                                      cv::Scalar(0,0,0),false);

        net.setInput(inputBlob,"data");
        cv::Mat prob = net.forward();
        int front = static_cast<int>(prob.at<float>(0,0)*FinedVertical.cols);
        int back = static_cast<int>(prob.at<float>(0,1)*FinedVertical.cols);
        front -= leftPadding ;
        if(front<0) front = 0;
        back +=rightPadding;
        if(back>FinedVertical.cols-1) back=FinedVertical.cols - 1;
        cv::Mat cropped  = FinedVertical.colRange(front,back).clone();
        return  cropped;


    }

    std::pair<int,int> FitLineRansac(std::vector<cv::Point> pts,int zeroadd = 0 ) {

        std::pair<int,int> res;
        if(pts.size()>2) {
            cv::Vec4f line;
            cv::fitLine(pts,line,cv::DIST_HUBER,0,0.01,0.01);
            float vx = line[0];
            float vy = line[1];
            float x = line[2];
            float y = line[3];
            int lefty = static_cast<int>((-x * vy / vx) + y);
            int righty = static_cast<int>(((136- x) * vy / vx) + y);
            res.first = lefty+PADDING_UP_DOWN+zeroadd;
            res.second = righty+PADDING_UP_DOWN+zeroadd;
            return res;
        }
        res.first = zeroadd;
        res.second = zeroadd;
        return res;
    }

    std::pair<int, int> FitLineRansac2(std::vector<cv::Point> pts,int width, int zeroadd = 0) {

        std::pair<int, int> res;
        if (pts.size()>2) {
            cv::Vec4f line;
            cv::fitLine(pts, line, cv::DIST_HUBER, 0, 0.01, 0.01);
            //cv::fitLine(pts, line, cv::DIST_L2, 0, 1e-2, 1e-2);
            float vx = line[0];
            float vy = line[1];
            float x = line[2];
            float y = line[3];
            int lefty = static_cast<int>((-x * vy / vx) + y);
            int righty = static_cast<int>(((width - x) * vy / vx) + y);
            res.first = lefty + PADDING_UP_DOWN + zeroadd;
            res.second = righty + PADDING_UP_DOWN + zeroadd;
            return res;
        }
        res.first = zeroadd;
        res.second = zeroadd;
        return res;
    }



    cv::Mat FineMapping::FineMappingVertical(cv::Mat InputProposal,int sliceNum,
                                             int upper,int lower,int windows_size){


        cv::Mat PreInputProposal;
        cv::Mat proposal;

        cv::resize(InputProposal,PreInputProposal,cv::Size(FINEMAPPING_W,FINEMAPPING_H));
        int x = InputProposal.channels();

        if(InputProposal.channels() == 3)
            cv::cvtColor(PreInputProposal,proposal,cv::COLOR_BGR2GRAY);
        else if(InputProposal.channels() == 4)
            cv::cvtColor(PreInputProposal,proposal,cv::COLOR_BGRA2GRAY);
        else
            PreInputProposal.copyTo(proposal);

//            proposal = PreInputProposal;

        // this will improve some sen
        cv::Mat kernal = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(1,3));
//        cv::erode(proposal,proposal,kernal);


        float diff = static_cast<float>(upper-lower);
        diff/=static_cast<float>(sliceNum-1);
        cv::Mat binary_adaptive;
        std::vector<cv::Point> line_upper;
        std::vector<cv::Point> line_lower;
        int contours_nums=0;

        for(int i = 0 ; i < sliceNum ; i++)
        {
            std::vector<std::vector<cv::Point> > contours;
            float k =lower + i*diff;
            cv::adaptiveThreshold(proposal,binary_adaptive,255,cv::ADAPTIVE_THRESH_MEAN_C,cv::THRESH_BINARY,windows_size,k);
            cv::Mat draw;
            binary_adaptive.copyTo(draw);
            cv::findContours(binary_adaptive,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
            for(auto contour: contours)
            {
                cv::Rect bdbox =cv::boundingRect(contour);
                float lwRatio = bdbox.height/static_cast<float>(bdbox.width);
                int  bdboxAera = bdbox.width*bdbox.height;
                if ((   lwRatio>0.7&&bdbox.width*bdbox.height>100 && bdboxAera<300)
                    || (lwRatio>3.0 && bdboxAera<100 && bdboxAera>10))
                {

                    cv::Point p1(bdbox.x, bdbox.y);
                    cv::Point p2(bdbox.x + bdbox.width, bdbox.y + bdbox.height);
                    line_upper.push_back(p1);
                    line_lower.push_back(p2);
                    contours_nums+=1;
                }
            }
        }

        std:: cout<<"contours_nums "<<contours_nums<<std::endl;

        if(contours_nums<41)
        {
            cv::bitwise_not(InputProposal,InputProposal);
            cv::Mat kernal = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(1,5));
            cv::Mat bak;
            cv::resize(InputProposal,bak,cv::Size(FINEMAPPING_W,FINEMAPPING_H));
            cv::erode(bak,bak,kernal);
            if(InputProposal.channels() == 3)
                cv::cvtColor(bak,proposal,cv::COLOR_BGR2GRAY);
            else
                proposal = bak;
            int contours_nums=0;

            for(int i = 0 ; i < sliceNum ; i++)
            {
                std::vector<std::vector<cv::Point> > contours;
                float k =lower + i*diff;
                cv::adaptiveThreshold(proposal,binary_adaptive,255,cv::ADAPTIVE_THRESH_MEAN_C,cv::THRESH_BINARY,windows_size,k);
//                cv::imshow("image",binary_adaptive);
//            cv::waitKey(0);
                cv::Mat draw;
                binary_adaptive.copyTo(draw);
                cv::findContours(binary_adaptive,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
                for(auto contour: contours)
                {
                    cv::Rect bdbox =cv::boundingRect(contour);
                    float lwRatio = bdbox.height/static_cast<float>(bdbox.width);
                    int  bdboxAera = bdbox.width*bdbox.height;
                    if ((   lwRatio>0.7&&bdbox.width*bdbox.height>120 && bdboxAera<300)
                        || (lwRatio>3.0 && bdboxAera<100 && bdboxAera>10))
                    {

                        cv::Point p1(bdbox.x, bdbox.y);
                        cv::Point p2(bdbox.x + bdbox.width, bdbox.y + bdbox.height);
                        line_upper.push_back(p1);
                        line_lower.push_back(p2);
                        contours_nums+=1;
                    }
                }
            }
//            std:: cout<<"contours_nums "<<contours_nums<<std::endl;
        }

            cv::Mat rgb;
            cv::copyMakeBorder(PreInputProposal, rgb, 30, 30, 0, 0, cv::BORDER_REPLICATE);
//        cv::imshow("rgb",rgb);
//        cv::waitKey(0);
//


            std::pair<int, int> A;
            std::pair<int, int> B;
            A = FitLineRansac(line_upper, -2);
            B = FitLineRansac(line_lower, 2);
            int leftyB = A.first;
            int rightyB = A.second;
            int leftyA = B.first;
            int rightyA = B.second;
            int cols = rgb.cols;
            int rows = rgb.rows;
//        pts_map1  = np.float32([[cols - 1, rightyA], [0, leftyA],[cols - 1, rightyB], [0, leftyB]])
//        pts_map2 = np.float32([[136,36],[0,36],[136,0],[0,0]])
//        mat = cv2.getPerspectiveTransform(pts_map1,pts_map2)
//        image = cv2.warpPerspective(rgb,mat,(136,36),flags=cv2.INTER_CUBIC)
            std::vector<cv::Point2f> corners(4);
            corners[0] = cv::Point2f(cols - 1, rightyA);
            corners[1] = cv::Point2f(0, leftyA);
            corners[2] = cv::Point2f(cols - 1, rightyB);
            corners[3] = cv::Point2f(0, leftyB);
            std::vector<cv::Point2f> corners_trans(4);
            corners_trans[0] = cv::Point2f(136, 36);
            corners_trans[1] = cv::Point2f(0, 36);
            corners_trans[2] = cv::Point2f(136, 0);
            corners_trans[3] = cv::Point2f(0, 0);
            cv::Mat transform = cv::getPerspectiveTransform(corners, corners_trans);
            cv::Mat quad = cv::Mat::zeros(36, 136, CV_8UC3);
            cv::warpPerspective(rgb, quad, transform, quad.size());
        return quad;

    }





    //斜率为负=左低右高，斜率为正=左高右低
    double getXieLv(cv::Point2f left_point, cv::Point2f right_point) {

        //直线的斜率
        std::vector<cv::Point2f> cn(2);

        //up
        cn[0] = right_point;//up-right
        cn[1] = left_point;//up-left

        //bot
        //cn[2] = cv::Point2f(cols - 1, B.second);//bot-right
        //cn[3] = cv::Point2f(0, B.first);//bot-left
        double k = (cn[1].y - cn[0].y) / (cn[1].x - cn[0].x + 0.000001);
        //不加0.000001 会变成曲线，斜率可能为0，即e.x-s.x可能为0
        double degree = atan(k) * 180 / 3.1415926;

        //double k2 = (cn[3].y - cn[2].y) / (cn[3].x - cn[2].x + 0.000001);
        //不加0.000001 会变成曲线，斜率可能为0，即e.x-s.x可能为0
        //double degree2 = atan(k2) * 180 / 3.1415926;

        //斜率为负=左低右高，斜率为正=左高右低

        //std::cout << "getRad bottom top " << degree2 << "||||||" << degree << std::endl;
        //std::cout << "line-up first second  " << A.first << "--" << A.second << std::endl;
        //std::cout << "line-bottom first second  " << B.first << "--" << B.second << std::endl;
        //end-------------------------

        //return degree;

        return k;

    }


    cv::Point getLeftTopPoint(std::vector<cv::Point> pts) {
        int tag = 10000;
        cv::Point pt;
        for (auto p : pts) {
            if (p.x < tag) {
                tag = p.x;
                pt = p;
            }
        }
        return pt;
    }

    cv::Point getLeftBottomPoint(std::vector<cv::Point> pts) {
        int tag = 10000;
        cv::Point pt;
        for (auto p : pts) {
            if (p.x < tag) {
                tag = p.x;
                pt = p;
            }
        }

        int tag2 = 0;
        cv::Point pt2;
        for (auto p : pts) {
            if (p.x == pt.x) {
                if (p.y > tag2) {
                    tag2 = p.y;
                    pt2 = p;
                }
            }

        }
        return pt2;
    }

    cv::Point getRightTopPoint(std::vector<cv::Point> pts) {
        int tag = 0;
        cv::Point pt;
        for (auto p : pts) {
            if (p.x > tag) {
                tag = p.x;
                pt = p;
            }
        }

        int tag2 = 10000;
        cv::Point pt2;
        for (auto p : pts) {
            if (p.x == pt.x) {
                if (p.y < tag2) {
                    tag2 = p.y;
                    pt2 = p;
                }
            }

        }
        return pt2;
    }

    cv::Point getRightBottomPoint(std::vector<cv::Point> pts) {
        int tag = 0;
        cv::Point pt;
        for (auto p : pts) {
            if (p.x > tag) {
                tag = p.x;
                pt = p;
            }
        }
        return pt;
    }

    //删除明显不符合条件的轮廓矩形
    std::vector<cv::Point> guolv2(std::vector<std::pair<cv::Point, cv::Rect>> its, cv::Rect rect) {

        std::vector<cv::Point> list;

        std::vector<std::pair<cv::Point, cv::Rect>> its2;
        //面积剔除
        int area2 = (int)(rect.width*rect.height)*0.8;
        int area3 = (int)(rect.width*rect.height)*0.2 + rect.width*rect.height;
        for (auto item : its) {
            int area = item.second.width*item.second.height;
            if (area > area2 && area<area3) {
                //list.push_back(item.first);
                its2.push_back(item);
            }
        }

        std::vector<std::pair<cv::Point, cv::Rect>> its3;
        //宽度剔除
        int width2 = rect.width*0.8;
        int width3 = rect.width*0.2+rect.width;
        for (auto item : its2) {
            if (item.second.width > width2 && item.second.width<width3) {
                //list.push_back(item.first);
                its3.push_back(item);
            }
        }

        //高度剔除
        int height2 = rect.height*0.8;
        int height3 = rect.height*0.2 + rect.height;
        for (auto item : its3) {
            if (item.second.height > height2 && item.second.height<height3) {
                list.push_back(item.first);
            }
        }


        return list;

    }

    std::vector<std::pair<cv::Point, cv::Rect>> guolv3(std::vector<std::pair<cv::Point, cv::Rect>> its,
                                                       cv::Rect rect) {

        std::vector<std::pair<cv::Point, cv::Rect>> its2;
        //面积剔除
        int area2 = (int)(rect.width*rect.height)*0.8;
        int area3 = (int)(rect.width*rect.height)*0.2 + rect.width*rect.height;
        for (auto item : its) {
            int area = item.second.width*item.second.height;
            if (area > area2 && area<area3) {
                its2.push_back(item);
            }
        }

        std::vector<std::pair<cv::Point, cv::Rect>> its3;
        //宽度剔除
        int width2 = rect.width*0.8;
        int width3 = rect.width*0.2 + rect.width;
        for (auto item : its2) {
            if (item.second.width > width2 && item.second.width<width3) {
                its3.push_back(item);
            }
        }

        std::vector<std::pair<cv::Point, cv::Rect>> its4;
        //高度剔除
        int height2 = rect.height*0.8;
        int height3 = rect.height*0.2 + rect.height;
        for (auto item : its3) {
            if (item.second.height > height2 && item.second.height<height3) {
                its4.push_back(item);
            }
        }

        return its4;

    }

    //删除重复的轮廓
    std::vector<std::pair<cv::Point, cv::Rect>> guolv3_2(std::vector<std::pair<cv::Point, cv::Rect>> its, cv::Rect rect) {

        std::vector<std::pair<cv::Point, cv::Rect>> its2;

        for (auto item : its) {
            cv::Rect r = item.second;
            int ct_x = (int)(r.x + r.x + r.width) / 2;

            int temp_span=10000;
            for (auto item2 : its2) {
                cv::Rect r2 = item2.second;
                int ct_x2 = (int)(r2.x + r2.x + r2.width) / 2;

                if (ct_x > ct_x2) {
                    temp_span = ct_x - ct_x2;
                }
                else {
                    temp_span = ct_x2 - ct_x;
                }
                if (temp_span < rect.width / 2) {
                    temp_span = 1;
                    break;
                }
                else {
                    temp_span = 2;
                }

            }

            if (temp_span == 10000 || temp_span==2) {
                its2.push_back(item);
            }

        }



        return its2;

    }




    //获取第三个矩形
    cv::Rect getNum3Rect(std::vector<std::pair<cv::Point, cv::Rect>> vector, int max_span) {

        cv::Rect rect_2;
        for (auto item : vector) {
            cv::Rect r = item.second;
            int ct_x = r.x;

            int temp_span;
            for (auto item2 : vector) {
                cv::Rect r2 = item2.second;
                int ct_x2 = r2.x;

                if(r.x!=r2.x){
                    if (ct_x > ct_x2) {
                        temp_span = ct_x - ct_x2-r2.width;
                        if (temp_span == max_span) {
                            rect_2 = r;
                            break;
                        }
                    }
                    else {
                        temp_span = ct_x2 - ct_x-r.width;
                        if (temp_span == max_span) {
                            rect_2 = r2;
                            break;
                        }
                    }
                }

            }

            if (rect_2.area() > 0) {
                break;
            }

        }

        return rect_2;

    }

    //获取集合中最右边的矩形
    cv::Rect getLastRightRect(std::vector<std::pair<cv::Point, cv::Rect>> vector) {

        cv::Rect rect_right;

        int tag = 0;
        for (auto item : vector) {
            cv::Rect r = item.second;
            if (r.x > tag) {
                tag = r.x;
                rect_right = r;
            }
        }

        return rect_right;

    }

    //获取车牌中字母之间的最大距离
    int getMaxSpan(std::vector<std::pair<cv::Point, cv::Rect>> vector) {

        int tag = 0;
        for (auto item : vector) {
            cv::Rect r = item.second;
            int temp_x = r.x;

            for (auto item2 : vector) {
                cv::Rect r2 = item2.second;
                int temp_x2 = r2.x;

                if (r.x != r2.x) {
                    int temp_span;
                    if (temp_x > temp_x2) {
                        temp_span = temp_x - temp_x2 - r2.width;
                    }
                    else {
                        temp_span = temp_x2 - temp_x - r.width;
                    }
                    if (temp_span > tag && temp_span<r2.width) {
                        tag = temp_span;
                    }
                }

            }
        }

        return tag;

    }

    //获取车牌中字母之间的最小距离
    int getMinSpan(std::vector<std::pair<cv::Point, cv::Rect>> vector, int max_span) {

        int tag = 100000;
        for (auto item : vector) {
            cv::Rect r = item.second;
            int temp_x = r.x;

            for (auto item2 : vector) {
                cv::Rect r2 = item2.second;
                int temp_x2 = r2.x;

                if (r.x != r2.x) {
                    int temp_span;
                    if (temp_x > temp_x2) {
                        temp_span = temp_x - temp_x2 - r2.width;
                    }
                    else {
                        temp_span = temp_x2 - temp_x - r.width;
                    }

                    if (temp_span < tag && temp_span<max_span) {
                        tag = temp_span;
                    }
                }

            }
        }

        return tag;
    }

    //获取车牌中平均的最小字母之间的距离
    int getMinSpanAve(std::vector<std::pair<cv::Point, cv::Rect>> vector,int max_span) {

        std::vector<int> arr;

        for (auto item : vector) {
            cv::Rect r = item.second;
            int temp_x = r.x;

            for (auto item2 : vector) {
                cv::Rect r2 = item2.second;
                int temp_x2 = r2.x;

                if (r.x != r2.x) {
                    int temp_span;
                    if (temp_x > temp_x2) {
                        temp_span = temp_x - temp_x2 - r2.width;
                    }
                    else {
                        temp_span = temp_x2 - temp_x - r.width;
                    }

                    if (temp_span<max_span) {
                        arr.push_back(temp_span);
                    }
                }

            }
        }

        int all = 0;
        for (auto item : arr) {
            all += item;
        }
        int tag = 0;
        if(arr.size()>0){
            tag = all / arr.size();
        }

        return tag;
    }


    //获得出现次数最多的面积的轮廓矩形
    cv::Rect getArea2(std::vector<std::pair<cv::Rect, int>> arr) {
        int k = 0;
        cv::Rect area;
        for (auto item : arr) {
            if (item.second > k) {
                k = item.second;
                area = item.first;
            }
        }
        return area;
    }


    //获得精准的中心的矩形
    cv::Rect getCenterRect(std::vector<std::pair<cv::Point, cv::Rect>> vector,
                           cv::Mat img,cv::Rect rect,double xielv) {

        cv::Rect cr;

        int max_span = getMaxSpan(vector);
        int min_span = getMinSpan(vector, max_span);
        int min_span_ave = getMinSpanAve(vector,max_span);


        int width_span = min_span_ave;
        int width_span2 = max_span;




        //方法一：如果集合的总数是7可推导出中心矩形；
        //方法二：如果存在第二个矩形可推导出中心矩形；
        cv::Rect rect_2;
        if (max_span > 2*min_span_ave) {
            rect_2 = getNum3Rect(vector, max_span);
        }

        if (rect_2.area() > 0) {

            int span = 10000;
            int ct_x_have = (int)(rect_2.x + rect_2.x + rect_2.width) / 2;
            int ct_x_temp = ct_x_have + rect_2.width + width_span;
            for (auto item : vector) {
                cv::Rect r = item.second;
                //当前小矩形的中心x坐标
                int ct_x_now = (int)(r.x + r.x + r.width) / 2;
                //推测的中心点和实际的中心点x坐标之差
                int k;
                if (ct_x_temp > ct_x_now) {
                    k = ct_x_temp - ct_x_now;
                }
                else {
                    k = ct_x_now - ct_x_temp;
                }
                if (k < span) {
                    span = k;
                    if (span < rect_2.width / 2) {
                        cr = item.second;
                    }
                }
            }
            if (cr.area() == 0) {
                double ddd = (rect_2.width + width_span)*fabs(xielv);
                int dd = ceil(ddd);
                if (xielv > 0) {
                    cr = cv::Rect(rect_2.x + rect_2.width + width_span, rect_2.y + dd, rect_2.width, rect_2.height);
                }
                else {
                    cr = cv::Rect(rect_2.x + rect_2.width + width_span, rect_2.y - dd, rect_2.width, rect_2.height);
                }
            }
            return cr;
        }






        //方法三：如果存在连续的5个右边的小矩形可推导出中心矩形；
        cv::Rect rect_right = getLastRightRect(vector);
        int k_add = 1;
        for(int i=0;i<4;i++){
            for (auto item : vector) {
                cv::Rect r = item.second;
                if (rect_right.x != r.x) {
                    int span = rect_right.x - r.x - r.width;
                    if (span >= min_span && span < max_span) {
                        rect_right = r;
                        k_add++;

                        if (k_add == 4) {
                            cr = r;
                        }
                        break;
                    }
                }
            }
        }
        if (k_add == 5) {
            return cr;
        }





        cv::Rect cr2;
        return cr2;

    }

    std::vector<cv::Rect> roll(std::vector<cv::Rect> list, int index,
                               int width_span, int width_span2,double xielv) {

        for (int i = index; i <=3; i++) {
            cv::Rect item = list[i];
            if (item.x == 0) {
                if (list[i + 1].x != 0) {

                    if (i == 1) {

                        double ddd = (list[i + 1].width + width_span2)*fabs(xielv);
                        int dd = ceil(ddd);

                        if (xielv > 0) {
                            list[i] = cv::Rect(list[i + 1].x - list[i + 1].width - width_span2, list[i + 1].y-dd,
                                               list[i + 1].width, list[i + 1].height);
                        }
                        else {
                            list[i] = cv::Rect(list[i + 1].x - list[i + 1].width - width_span2, list[i + 1].y+dd,
                                               list[i + 1].width, list[i + 1].height);
                        }


                    }
                    else {

                        double ddd = (list[i + 1].width+width_span)*fabs(xielv);
                        int dd = ceil(ddd);
                        //std::cout << "ddd:"<<list[i+1].width<<"--"<< ddd << std::endl;

                        if (xielv > 0) {
                            list[i] = cv::Rect(list[i + 1].x - list[i + 1].width - width_span, list[i + 1].y-dd,
                                               list[i + 1].width, list[i + 1].height);
                        }
                        else {
                            list[i] = cv::Rect(list[i + 1].x - list[i + 1].width - width_span, list[i + 1].y+dd,
                                               list[i + 1].width, list[i + 1].height);
                        }


                    }

                }
                else {
                    roll(list, i + 1, width_span, width_span2,xielv);
                }
            }
        }

        return list;

    }

    std::vector<cv::Rect> roll2(std::vector<cv::Rect> list, int index,
                                int width_span,double xielv) {

        for (int i = index; i >= 3; i--) {
            cv::Rect item = list[i];
            if (item.x == 0) {
                if (list[i - 1].x != 0) {

                    double ddd = (list[i - 1].width + width_span)*fabs(xielv);
                    int dd = ceil(ddd);

                    if (xielv > 0) {

                        list[i] = cv::Rect(list[i - 1].x + list[i - 1].width + width_span, list[i - 1].y + dd, list[i - 1].width, list[i - 1].height);
                    }
                    else {

                        list[i] = cv::Rect(list[i - 1].x + list[i - 1].width + width_span, list[i - 1].y-dd,list[i - 1].width, list[i - 1].height);
                    }

                }
                else {
                    roll2(list, i - 1, width_span,xielv);
                }
            }
        }

        return list;

    }

    //根据中心矩形推导出其它的矩形一个一个的推导
    std::vector<cv::Rect> getRects2(cv::Mat img,
                                    cv::Rect center_rect,
                                    std::vector<std::pair<cv::Point, cv::Rect>> list,
                                    double xielv) {

        std::vector<cv::Rect> list2(7);

        int width_img = img.cols;
        int scale = (int)width_img / 140;
        //矩形之接的距离
        int width_span = 4 * scale;
        int width_span2 = 4 * 3 * scale;

        //中心矩形的中心x坐标
        int center_point_x_in_cr = (int)(center_rect.x + center_rect.x + center_rect.width) / 2;

        //待寻找位置的中心位置
        int temp_center_x;

        for (int i = 0; i < 7; i++) {

            if (i == 3) {
                list2[3] = center_rect;
                continue;
            }

            int weizhi = i;

            if (weizhi > 3) {
                //右边矩形

                //相差间隔数
                int span = weizhi - 3;
                temp_center_x = center_point_x_in_cr + span*(center_rect.width + width_span);

            }
            else {
                //左边矩形

                //相差间隔数
                int span = 3 - weizhi;

                if (weizhi == 2) {
                    temp_center_x = center_point_x_in_cr - span*(center_rect.width + width_span);
                }
                else {
                    temp_center_x = center_point_x_in_cr - (span - 1)*(center_rect.width + width_span) - center_rect.width - width_span2;
                }

            }

            int span = 10000;

            for (auto item : list) {
                cv::Rect r = item.second;
                int center_point_x_in_rect = (int)(r.x + r.x + r.width) / 2;

                int k;
                if (temp_center_x > center_point_x_in_rect) {
                    k = temp_center_x - center_point_x_in_rect;
                }
                else {
                    k = center_point_x_in_rect - temp_center_x;
                }

                if (k < span) {
                    span = k;
                    if (span < center_rect.width / 2) {
                        list2[weizhi] = item.second;
                    }
                }

            }


        }


        for (int i = 0; i < 3; i++) {
            cv::Rect item = list2[i];
            if (item.x == 0) {
                list2 = roll(list2, i, width_span, width_span2,xielv);
            }
        }
        for (int i = 0; i < 3; i++) {
            cv::Rect item = list2[i];
            if (item.x == 0) {
                list2 = roll(list2, i, width_span, width_span2, xielv);
            }
        }
        for (int i = 0; i < 3; i++) {
            cv::Rect item = list2[i];
            if (item.x == 0) {
                list2 = roll(list2, i, width_span, width_span2, xielv);
            }
        }





        for (int i = list2.size() - 1;i>3; i--) {
            cv::Rect item = list2[i];
            if (item.x == 0) {
                list2 = roll2(list2, i, width_span,xielv);
            }
        }
        for (int i = list2.size() - 1; i>3; i--) {
            cv::Rect item = list2[i];
            if (item.x == 0) {
                list2 = roll2(list2, i, width_span, xielv);
            }
        }
        for (int i = list2.size() - 1; i>3; i--) {
            cv::Rect item = list2[i];
            if (item.x == 0) {
                list2 = roll2(list2, i, width_span, xielv);
            }
        }


        return list2;

    }

    //根据中心矩形获取全部的小矩形，精准无推导
    std::pair<int,std::vector<cv::Rect>> getRects3(cv::Mat img, cv::Rect center_rect,
                                                   std::vector<std::pair<cv::Point, cv::Rect>> list,
                                                   double xielv) {

        std::pair<int, std::vector<cv::Rect>> pair;

        std::vector<cv::Rect> list2(7);

        list2[3] = center_rect;

        int width_img = img.cols;
        int scale = (int)width_img / 140;
        //矩形之接的距离
        int width_span = 4 * scale;
        int width_span2 = 4 * 3 * scale;



        int span = 10000;
        int ii = 2;
        int ct_x_have = (int)(list2[ii+1].x + list2[ii+1].x + list2[ii+1].width) / 2;
        int ct_x_temp = ct_x_have - list2[ii + 1].width - width_span;
        for (auto item : list) {
            cv::Rect r = item.second;
            //当前小矩形的中心x坐标
            int ct_x_now = (int)(r.x + r.x + r.width) / 2;
            //推测的中心点和实际的中心点x坐标之差
            int k;
            if (ct_x_temp > ct_x_now) {
                k = ct_x_temp - ct_x_now;
            }
            else {
                k = ct_x_now - ct_x_temp;
            }
            if (k < span) {
                span = k;
                if (span < list2[ii+1].width / 2) {
                    list2[ii] = item.second;
                }
            }
        }
        if (list2[ii].area() == 0) {
            pair.first = 0;
            return pair;
        }

        span = 10000;
        ii = 1;
        ct_x_have = (int)(list2[ii+1].x + list2[ii+1].x + list2[ii+1].width) / 2;
        ct_x_temp = ct_x_have - list2[ii+1].width - width_span2;
        for (auto item : list) {
            cv::Rect r = item.second;
            //当前小矩形的中心x坐标
            int ct_x_now = (int)(r.x + r.x + r.width) / 2;
            //推测的中心点和实际的中心点x坐标之差
            int k;
            if (ct_x_temp > ct_x_now) {
                k = ct_x_temp - ct_x_now;
            }
            else {
                k = ct_x_now - ct_x_temp;
            }
            if (k < span) {
                span = k;
                if (span < list2[ii+1].width / 2) {
                    list2[ii] = item.second;
                }
            }
        }
        if (list2[ii].area() == 0) {
            pair.first = 0;
            return pair;
        }


        span = 10000;
        ii = 0;
        ct_x_have = (int)(list2[ii + 1].x + list2[ii + 1].x + list2[ii + 1].width) / 2;
        ct_x_temp = ct_x_have - list2[ii + 1].width - width_span;
        for (auto item : list) {
            cv::Rect r = item.second;
            //当前小矩形的中心x坐标
            int ct_x_now = (int)(r.x + r.x + r.width) / 2;
            //推测的中心点和实际的中心点x坐标之差
            int k;
            if (ct_x_temp > ct_x_now) {
                k = ct_x_temp - ct_x_now;
            }
            else {
                k = ct_x_now - ct_x_temp;
            }
            if (k < span) {
                span = k;
                if (span < list2[ii + 1].width / 2) {
                    list2[ii] = item.second;
                }
            }
        }
        if (list2[ii].area() == 0) {
            pair.first = 0;
            return pair;
        }

        //--------------------------------------------------

        span = 10000;
        ii = 4;
        ct_x_have = (int)(list2[ii - 1].x + list2[ii - 1].x + list2[ii - 1].width) / 2;
        ct_x_temp = ct_x_have + list2[ii - 1].width + width_span;
        for (auto item : list) {
            cv::Rect r = item.second;
            //当前小矩形的中心x坐标
            int ct_x_now = (int)(r.x + r.x + r.width) / 2;
            //推测的中心点和实际的中心点x坐标之差
            int k;
            if (ct_x_temp > ct_x_now) {
                k = ct_x_temp - ct_x_now;
            }
            else {
                k = ct_x_now - ct_x_temp;
            }
            if (k < span) {
                span = k;
                if (span < list2[ii - 1].width / 2) {
                    list2[ii] = item.second;
                }
            }
        }
        if (list2[ii].area() == 0) {
            pair.first = 0;
            return pair;
        }


        span = 10000;
        ii = 5;
        ct_x_have = (int)(list2[ii - 1].x + list2[ii - 1].x + list2[ii - 1].width) / 2;
        ct_x_temp = ct_x_have + list2[ii - 1].width + width_span;
        for (auto item : list) {
            cv::Rect r = item.second;
            //当前小矩形的中心x坐标
            int ct_x_now = (int)(r.x + r.x + r.width) / 2;
            //推测的中心点和实际的中心点x坐标之差
            int k;
            if (ct_x_temp > ct_x_now) {
                k = ct_x_temp - ct_x_now;
            }
            else {
                k = ct_x_now - ct_x_temp;
            }
            if (k < span) {
                span = k;
                if (span < list2[ii - 1].width / 2) {
                    list2[ii] = item.second;
                }
            }
        }
        if (list2[ii].area() == 0) {
            pair.first = 0;
            return pair;
        }


        span = 10000;
        ii = 6;
        ct_x_have = (int)(list2[ii - 1].x + list2[ii - 1].x + list2[ii - 1].width) / 2;
        ct_x_temp = ct_x_have + list2[ii - 1].width + width_span;
        for (auto item : list) {
            cv::Rect r = item.second;
            //当前小矩形的中心x坐标
            int ct_x_now = (int)(r.x + r.x + r.width) / 2;
            //推测的中心点和实际的中心点x坐标之差
            int k;
            if (ct_x_temp > ct_x_now) {
                k = ct_x_temp - ct_x_now;
            }
            else {
                k = ct_x_now - ct_x_temp;
            }
            if (k < span) {
                span = k;
                if (span < list2[ii - 1].width / 2) {
                    list2[ii] = item.second;
                }
            }
        }
        if (list2[ii].area() == 0) {
            pair.first = 0;
            return pair;
        }


        pair.first = 1;
        pair.second = list2;

        return pair;

    }

    //根据中心矩形获取全部的小矩形，加推导
    std::pair<int,std::vector<cv::Rect>> getRects4(cv::Mat img, cv::Rect center_rect,
                                                   std::vector<std::pair<cv::Point, cv::Rect>> list,
                                                   double xielv) {

        std::pair<int, std::vector<cv::Rect>> pair;

        std::vector<cv::Rect> list2(7);

        list2[3] = center_rect;

        int width_img = img.cols;
        int scale = (int)width_img / 140;
        //矩形之接的距离
        int width_span = 4 * scale;
        int width_span2 = 4 * 3 * scale;



        int span = 10000;
        int ii = 2;
        int ct_x_have = (int)(list2[ii + 1].x + list2[ii + 1].x + list2[ii + 1].width) / 2;
        int ct_x_temp = ct_x_have - list2[ii + 1].width - width_span;
        for (auto item : list) {
            cv::Rect r = item.second;
            //当前小矩形的中心x坐标
            int ct_x_now = (int)(r.x + r.x + r.width) / 2;
            //推测的中心点和实际的中心点x坐标之差
            int k;
            if (ct_x_temp > ct_x_now) {
                k = ct_x_temp - ct_x_now;
            }
            else {
                k = ct_x_now - ct_x_temp;
            }
            if (k < span) {
                span = k;
                if (span < list2[ii + 1].width / 2) {
                    list2[ii] = item.second;
                }
            }
        }
        if (list2[ii].area() == 0) {
            double ddd = (list2[ii + 1].width + width_span)*fabs(xielv);
            int dd = ceil(ddd);
            if (xielv > 0) {
                list2[ii] = cv::Rect(list2[ii + 1].x - list2[ii + 1].width - width_span, list2[ii + 1].y - dd,
                                     list2[ii + 1].width, list2[ii + 1].height);
            }
            else {
                list2[ii] = cv::Rect(list2[ii + 1].x - list2[ii + 1].width - width_span, list2[ii + 1].y + dd,
                                     list2[ii + 1].width, list2[ii + 1].height);
            }
        }


        span = 10000;
        ii = 1;
        ct_x_have = (int)(list2[ii + 1].x + list2[ii + 1].x + list2[ii + 1].width) / 2;
        ct_x_temp = ct_x_have - list2[ii + 1].width - width_span2;
        for (auto item : list) {
            cv::Rect r = item.second;
            //当前小矩形的中心x坐标
            int ct_x_now = (int)(r.x + r.x + r.width) / 2;
            //推测的中心点和实际的中心点x坐标之差
            int k;
            if (ct_x_temp > ct_x_now) {
                k = ct_x_temp - ct_x_now;
            }
            else {
                k = ct_x_now - ct_x_temp;
            }
            if (k < span) {
                span = k;
                if (span < list2[ii + 1].width / 2) {
                    list2[ii] = item.second;
                }
            }
        }
        if (list2[ii].area() == 0) {
            double ddd = (list2[ii + 1].width + width_span2)*fabs(xielv);
            int dd = ceil(ddd);
            if (xielv > 0) {
                list2[ii] = cv::Rect(list2[ii + 1].x - list2[ii + 1].width - width_span, list2[ii + 1].y - dd,
                                     list2[ii + 1].width, list2[ii + 1].height);
            }
            else {
                list2[ii] = cv::Rect(list2[ii + 1].x - list2[ii + 1].width - width_span, list2[ii + 1].y + dd,
                                     list2[ii + 1].width, list2[ii + 1].height);
            }
        }


        span = 10000;
        ii = 0;
        ct_x_have = (int)(list2[ii + 1].x + list2[ii + 1].x + list2[ii + 1].width) / 2;
        ct_x_temp = ct_x_have - list2[ii + 1].width - width_span;
        for (auto item : list) {
            cv::Rect r = item.second;
            //当前小矩形的中心x坐标
            int ct_x_now = (int)(r.x + r.x + r.width) / 2;
            //推测的中心点和实际的中心点x坐标之差
            int k;
            if (ct_x_temp > ct_x_now) {
                k = ct_x_temp - ct_x_now;
            }
            else {
                k = ct_x_now - ct_x_temp;
            }
            if (k < span) {
                span = k;
                if (span < list2[ii + 1].width / 2) {
                    list2[ii] = item.second;
                }
            }
        }
        if (list2[ii].area() == 0) {
            double ddd = (list2[ii + 1].width + width_span)*fabs(xielv);
            int dd = ceil(ddd);
            if (xielv > 0) {
                list2[ii] = cv::Rect(list2[ii + 1].x - list2[ii + 1].width - width_span, list2[ii + 1].y - dd,
                                     list2[ii + 1].width, list2[ii + 1].height);
            }
            else {
                list2[ii] = cv::Rect(list2[ii + 1].x - list2[ii + 1].width - width_span, list2[ii + 1].y + dd,
                                     list2[ii + 1].width, list2[ii + 1].height);
            }
        }

        //--------------------------------------------------

        span = 10000;
        ii = 4;
        ct_x_have = (int)(list2[ii - 1].x + list2[ii - 1].x + list2[ii - 1].width) / 2;
        ct_x_temp = ct_x_have + list2[ii - 1].width + width_span;
        for (auto item : list) {
            cv::Rect r = item.second;
            //当前小矩形的中心x坐标
            int ct_x_now = (int)(r.x + r.x + r.width) / 2;
            //推测的中心点和实际的中心点x坐标之差
            int k;
            if (ct_x_temp > ct_x_now) {
                k = ct_x_temp - ct_x_now;
            }
            else {
                k = ct_x_now - ct_x_temp;
            }
            if (k < span) {
                span = k;
                if (span < list2[ii - 1].width / 2) {
                    list2[ii] = item.second;
                }
            }
        }
        if (list2[ii].area() == 0) {
            double ddd = (list2[ii - 1].width + width_span)*fabs(xielv);
            int dd = ceil(ddd);
            if (xielv > 0) {
                list2[ii] = cv::Rect(list2[ii - 1].x + list2[ii - 1].width + width_span, list2[ii - 1].y + dd, list2[ii - 1].width, list2[ii - 1].height);
            }
            else {

                list2[ii] = cv::Rect(list2[ii - 1].x + list2[ii - 1].width + width_span, list2[ii - 1].y - dd, list2[ii - 1].width, list2[ii - 1].height);
            }
        }


        span = 10000;
        ii = 5;
        ct_x_have = (int)(list2[ii - 1].x + list2[ii - 1].x + list2[ii - 1].width) / 2;
        ct_x_temp = ct_x_have + list2[ii - 1].width + width_span;
        for (auto item : list) {
            cv::Rect r = item.second;
            //当前小矩形的中心x坐标
            int ct_x_now = (int)(r.x + r.x + r.width) / 2;
            //推测的中心点和实际的中心点x坐标之差
            int k;
            if (ct_x_temp > ct_x_now) {
                k = ct_x_temp - ct_x_now;
            }
            else {
                k = ct_x_now - ct_x_temp;
            }
            if (k < span) {
                span = k;
                if (span < list2[ii - 1].width / 2) {
                    list2[ii] = item.second;
                }
            }
        }
        if (list2[ii].area() == 0) {
            double ddd = (list2[ii - 1].width + width_span)*fabs(xielv);
            int dd = ceil(ddd);
            if (xielv > 0) {
                list2[ii] = cv::Rect(list2[ii - 1].x + list2[ii - 1].width + width_span, list2[ii - 1].y + dd, list2[ii - 1].width, list2[ii - 1].height);
            }
            else {

                list2[ii] = cv::Rect(list2[ii - 1].x + list2[ii - 1].width + width_span, list2[ii - 1].y - dd, list2[ii - 1].width, list2[ii - 1].height);
            }
        }


        span = 10000;
        ii = 6;
        ct_x_have = (int)(list2[ii - 1].x + list2[ii - 1].x + list2[ii - 1].width) / 2;
        ct_x_temp = ct_x_have + list2[ii - 1].width + width_span;
        for (auto item : list) {
            cv::Rect r = item.second;
            //当前小矩形的中心x坐标
            int ct_x_now = (int)(r.x + r.x + r.width) / 2;
            //推测的中心点和实际的中心点x坐标之差
            int k;
            if (ct_x_temp > ct_x_now) {
                k = ct_x_temp - ct_x_now;
            }
            else {
                k = ct_x_now - ct_x_temp;
            }
            if (k < span) {
                span = k;
                if (span < list2[ii - 1].width / 2) {
                    list2[ii] = item.second;
                }
            }
        }
        if (list2[ii].area() == 0) {
            double ddd = (list2[ii - 1].width + width_span)*fabs(xielv);
            int dd = ceil(ddd);
            if (xielv > 0) {
                list2[ii] = cv::Rect(list2[ii - 1].x + list2[ii - 1].width + width_span, list2[ii - 1].y + dd, list2[ii - 1].width, list2[ii - 1].height);
            }
            else {

                list2[ii] = cv::Rect(list2[ii - 1].x + list2[ii - 1].width + width_span, list2[ii - 1].y - dd, list2[ii - 1].width, list2[ii - 1].height);
            }
        }




        int k = 0;
        for (int i = 0; i < 7; i++) {
            if (list2[i].area()>0) {

            }
            else {
                k = 1;
            }
        }

        if (k == 1) {
            pair.first = 0;
        }
        else {
            pair.first = 1;
            pair.second = list2;
        }

        return pair;

    }

    //根据中心矩形获取全部的小矩形，加推导，放大每个小矩形
    std::pair<int, std::vector<cv::Rect>> getRects5(cv::Mat img,
                                                    cv::Rect center_rect,
                                                    std::vector<std::pair<cv::Point, cv::Rect>> list,
                                                    double xielv) {

        std::pair<int, std::vector<cv::Rect>> pair;

        std::vector<cv::Rect> list2(7);
        list2[3] = center_rect;

        int max_span = getMaxSpan(list);
        int min_span = getMinSpan(list, max_span);
        int min_span_ave = getMinSpanAve(list,max_span);


        int width_span = min_span_ave;
        int width_span2 = max_span;

        int width_img = img.cols;
        float scale = width_img / 140.00;

//        //矩形之接的距离
//        int width_span = 4 * scale;
//        int width_span2 = 4 * 3 * scale;

        int width_outline = 2 * scale;
        //width_outline = 0;



        int span = 10000;
        int ii = 2;
        int ct_x_temp = list2[ii + 1].x - list2[ii + 1].width - width_span;
        for (auto item : list) {
            cv::Rect r = item.second;
            int ct_x_now = r.x;

            int k;
            if (ct_x_temp > ct_x_now) {
                k = ct_x_temp - ct_x_now;
            }
            else {
                k = ct_x_now - ct_x_temp;
            }
            if (k < span) {
                span = k;
                if (span < list2[ii + 1].width / 2) {
                    list2[ii] = item.second;
                }
            }
        }
        if (list2[ii].area() == 0) {
            double ddd = (list2[ii + 1].width + width_span)*fabs(xielv);
            int dd = ceil(ddd);
            if (xielv > 0) {
                list2[ii] = cv::Rect(list2[ii + 1].x - list2[ii + 1].width - width_span, list2[ii + 1].y - dd,
                                     list2[ii + 1].width, list2[ii + 1].height);
            }
            else {
                list2[ii] = cv::Rect(list2[ii + 1].x - list2[ii + 1].width - width_span, list2[ii + 1].y + dd,
                                     list2[ii + 1].width, list2[ii + 1].height);
            }
        }


        span = 10000;
        ii = 1;
        ct_x_temp = list2[ii + 1].x - list2[ii + 1].width - width_span2;
        for (auto item : list) {
            cv::Rect r = item.second;
            int ct_x_now = r.x;

            int k;
            if (ct_x_temp > ct_x_now) {
                k = ct_x_temp - ct_x_now;
            }
            else {
                k = ct_x_now - ct_x_temp;
            }
            if (k < span) {
                span = k;
                if (span < list2[ii + 1].width / 2) {
                    list2[ii] = item.second;
                }
            }
        }
        if (list2[ii].area() == 0) {
            double ddd = (list2[ii + 1].width + width_span2)*fabs(xielv);
            int dd = ceil(ddd);
            if (xielv > 0) {
                list2[ii] = cv::Rect(list2[ii + 1].x - list2[ii + 1].width - width_span, list2[ii + 1].y - dd,
                                     list2[ii + 1].width, list2[ii + 1].height);
            }
            else {
                list2[ii] = cv::Rect(list2[ii + 1].x - list2[ii + 1].width - width_span, list2[ii + 1].y + dd,
                                     list2[ii + 1].width, list2[ii + 1].height);
            }
        }


        span = 10000;
        ii = 0;
        ct_x_temp = list2[ii + 1].x - list2[ii + 1].width - width_span;
        for (auto item : list) {
            cv::Rect r = item.second;
            int ct_x_now = r.x;

            int k;
            if (ct_x_temp > ct_x_now) {
                k = ct_x_temp - ct_x_now;
            }
            else {
                k = ct_x_now - ct_x_temp;
            }
            if (k < span) {
                span = k;
                if (span < list2[ii + 1].width / 2) {
                    list2[ii] = item.second;
                }
            }
        }
        if (list2[ii].area() == 0) {
            double ddd = (list2[ii + 1].width + width_span)*fabs(xielv);
            int dd = ceil(ddd);
            if (xielv > 0) {
                list2[ii] = cv::Rect(list2[ii + 1].x - list2[ii + 1].width - width_span, list2[ii + 1].y - dd,
                                     list2[ii + 1].width, list2[ii + 1].height);
            }
            else {
                list2[ii] = cv::Rect(list2[ii + 1].x - list2[ii + 1].width - width_span, list2[ii + 1].y + dd,
                                     list2[ii + 1].width, list2[ii + 1].height);
            }
        }

        //--------------------------------------------------

        span = 10000;
        ii = 4;
        ct_x_temp = list2[ii - 1].x + list2[ii - 1].width + width_span;
        for (auto item : list) {
            cv::Rect r = item.second;
            int ct_x_now = r.x;

            int k;
            if (ct_x_temp > ct_x_now) {
                k = ct_x_temp - ct_x_now;
            }
            else {
                k = ct_x_now - ct_x_temp;
            }
            if (k < span) {
                span = k;
                if (span < list2[ii - 1].width / 2) {
                    list2[ii] = item.second;
                }
            }
        }
        if (list2[ii].area() == 0) {
            double ddd = (list2[ii - 1].width + width_span)*fabs(xielv);
            int dd = ceil(ddd);
            if (xielv > 0) {
                list2[ii] = cv::Rect(list2[ii - 1].x + list2[ii - 1].width + width_span, list2[ii - 1].y + dd, list2[ii - 1].width, list2[ii - 1].height);
            }
            else {

                list2[ii] = cv::Rect(list2[ii - 1].x + list2[ii - 1].width + width_span, list2[ii - 1].y - dd, list2[ii - 1].width, list2[ii - 1].height);
            }
        }


        span = 10000;
        ii = 5;
        ct_x_temp = list2[ii - 1].x + list2[ii - 1].width + width_span;
        for (auto item : list) {
            cv::Rect r = item.second;
            int ct_x_now = r.x;

            int k;
            if (ct_x_temp > ct_x_now) {
                k = ct_x_temp - ct_x_now;
            }
            else {
                k = ct_x_now - ct_x_temp;
            }
            if (k < span) {
                span = k;
                if (span < list2[ii - 1].width / 2) {
                    list2[ii] = item.second;
                }
            }
        }
        if (list2[ii].area() == 0) {
            double ddd = (list2[ii - 1].width + width_span)*fabs(xielv);
            int dd = ceil(ddd);
            if (xielv > 0) {
                list2[ii] = cv::Rect(list2[ii - 1].x + list2[ii - 1].width + width_span, list2[ii - 1].y + dd, list2[ii - 1].width, list2[ii - 1].height);
            }
            else {

                list2[ii] = cv::Rect(list2[ii - 1].x + list2[ii - 1].width + width_span, list2[ii - 1].y - dd, list2[ii - 1].width, list2[ii - 1].height);
            }
        }


        span = 10000;
        ii = 6;
        ct_x_temp = list2[ii - 1].x + list2[ii - 1].width + width_span;
        for (auto item : list) {
            cv::Rect r = item.second;
            int ct_x_now = r.x;

            int k;
            if (ct_x_temp > ct_x_now) {
                k = ct_x_temp - ct_x_now;
            }
            else {
                k = ct_x_now - ct_x_temp;
            }
            if (k < span) {
                span = k;
                if (span < list2[ii - 1].width / 2) {
                    list2[ii] = item.second;
                }
            }
        }
        if (list2[ii].area() == 0) {
            double ddd = (list2[ii - 1].width + width_span)*fabs(xielv);
            int dd = ceil(ddd);
            if (xielv > 0) {
                list2[ii] = cv::Rect(list2[ii - 1].x + list2[ii - 1].width + width_span, list2[ii - 1].y + dd, list2[ii - 1].width, list2[ii - 1].height);
            }
            else {
                list2[ii] = cv::Rect(list2[ii - 1].x + list2[ii - 1].width + width_span, list2[ii - 1].y - dd, list2[ii - 1].width, list2[ii - 1].height);
            }
        }


        //放大小矩形
        for (int i = 0; i < 7; i++) {
            cv::Rect r = list2[i];

            r.x = r.x - width_outline;
            r.y = r.y - width_outline;
            r.width = r.width + 2 * width_outline;
            r.height = r.height + 2 * width_outline;

            //防止超出边界
            if (r.x < 0) {
                r.x = 0;
            }
            if (r.y < 0) {
                r.y = 0;
            }
            if (r.width + r.x >= img.cols) {
                r.width = img.cols - r.x;
            }
            if (r.height + r.y >= img.rows) {
                r.height = img.rows - r.y;
            }
            if(r.width<0){
                r.width = 0;
            }
            if(r.height<0){
                r.height = 0;
            }

            list2[i] = r;

        }


        pair.first = 1;
        pair.second = list2;

        return pair;

    }






    //获得图片的中心点
    cv::Point getCenterPoint(cv::Mat img) {

        int width_img = img.cols;
        float scale = width_img / 140;
        //int span = (int)5 * scale;


        cv::Point cp;

        int img_width_half = (int)img.cols / 2;
        int img_height_half = (int)img.rows / 2;

        cp.x = img_width_half;
        cp.y = img_height_half;

        return cp;
    }
    //获得中心的矩形
    cv::Rect getCenterRect(cv::Rect rect,cv::Point p) {

        cv::Rect r(0,0,0,0);

        int center_point_x_in_rect = (int)(2 * rect.x + rect.width) / 2;

        int span;

        if (center_point_x_in_rect > p.x) {
            span = center_point_x_in_rect - p.x;
        }
        else {
            span = p.x - center_point_x_in_rect;
        }

        if (span < rect.width/2) {
            r = rect;
        }

        return r;
    }
    //获得最接近中心的矩形
    cv::Rect getWillCenterRect(std::vector<std::pair<cv::Point, cv::Rect>> vector,cv::Mat img) {

        cv::Rect cr;

        int img_width_half = (int)img.cols/2;

        int span = 10000;

        for (auto item : vector) {
            cv::Rect r = item.second;
            int center_point_x_in_rect = (int)(2*r.x + r.width) / 2;

            int k;
            if (img_width_half > center_point_x_in_rect) {
                k = img_width_half - center_point_x_in_rect;
            }
            else {
                k = center_point_x_in_rect - img_width_half;
            }

            //std::cout << "lll:" << r.x << "--" << r.width << "--" << center_point_x_in_rect<<"-"<<k << std::endl;

            if (k < span) {
                span = k;
                cr = item.second;
            }

        }


        return cr;

    }

    //获得车牌4个拐点
    std::vector<cv::Point> FineMapping::get4point(cv::Mat InputProposal,
                                                  int sliceNum, int upper,
                                                  int lower, int windows_size) {

        cv::Mat proposal;

        cv::Mat c, c2, c3;
        InputProposal.copyTo(c);
        InputProposal.copyTo(c2);
        InputProposal.copyTo(c3);

        if (InputProposal.channels() == 3) {
            cv::cvtColor(InputProposal, proposal, cv::COLOR_BGR2GRAY);
        } else {
            InputProposal.copyTo(proposal);
        }

        float diff = static_cast<float>(upper - lower);
        diff /= static_cast<float>(sliceNum - 1);
        cv::Mat binary_adaptive;
        std::vector<cv::Point> line_upper;
        std::vector<cv::Point> line_lower;

        std::vector<cv::Point> line_upper_left_top;
        std::vector<cv::Point> line_lower_right_bottom;
        std::vector<cv::Point> line_upper_right_top;
        std::vector<cv::Point> line_lower_left_bottom;

        for (int i = 0; i < sliceNum; i++) {

            std::vector<std::vector<cv::Point>> contours;

            float k = lower + i*diff;

            //自适应二值化
            cv::adaptiveThreshold(proposal, binary_adaptive,
                                  255,
                                  cv::ADAPTIVE_THRESH_MEAN_C,
                                  cv::THRESH_BINARY,
                                  windows_size, k);

            //找轮廓
            cv::findContours(binary_adaptive, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            for (auto contour : contours) {

                //根据轮廓获取最小的矩形
                cv::Rect bdbox = cv::boundingRect(contour);

                //矩形的高/宽
                float lwRatio = bdbox.height / static_cast<float>(bdbox.width);
                //矩形的面积
                int  bdboxAera = bdbox.width*bdbox.height;

                if (lwRatio > 1 && bdboxAera > 100) {

                    cv::Point p1(bdbox.x, bdbox.y - 5);//left-top
                    cv::Point p2(bdbox.x + bdbox.width, bdbox.y + bdbox.height + 5);//right-bottom

                    line_upper_left_top.push_back(p1);
                    line_lower_right_bottom.push_back(p2);

                    cv::Point p3(bdbox.x + bdbox.width, bdbox.y - 5);//right-top
                    cv::Point p4(bdbox.x, bdbox.y + bdbox.height + 5);//left-bottom

                    line_upper_right_top.push_back(p3);
                    line_lower_left_bottom.push_back(p4);

                }


            }

        }

        cv::Mat rgb;
        //给图片增加边距
        cv::copyMakeBorder(InputProposal,
                           rgb,
                           PADDING_UP_DOWN,
                           PADDING_UP_DOWN,
                           0, 0, cv::BORDER_REPLICATE);

        std::pair<int, int> A;
        std::pair<int, int> B;

        A = FitLineRansac2(line_upper_right_top, rgb.cols, -1);
        B = FitLineRansac2(line_lower_left_bottom, rgb.cols, 1);
        int leftyB = A.first;
        int rightyB = A.second;
        int leftyA = B.first;
        int rightyA = B.second;
        int cols = rgb.cols;
        int rows = rgb.rows;

        cv::Point2f p1 = cv::Point2f(0, A.first);
        cv::Point2f p2 = cv::Point2f(cols - 1, A.second);
        double dg = getXieLv(p1, p2);

        if (dg < 0) {
            A = FitLineRansac2(line_upper_left_top, rgb.cols, -1);
            B = FitLineRansac2(line_lower_right_bottom, rgb.cols, 1);
            leftyB = A.first;
            rightyB = A.second;
            leftyA = B.first;
            rightyA = B.second;
            cols = rgb.cols;
            rows = rgb.rows;
        }


        std::vector<cv::Point2f> corners(4);
        corners[0] = cv::Point2f(cols - 1, rightyA);//bottom-right
        corners[1] = cv::Point2f(0, leftyA);//bottom-left
        corners[2] = cv::Point2f(cols - 1, rightyB);//top-right
        corners[3] = cv::Point2f(0, leftyB);//top-left

        //遮去无效部分
        cv::Point root_points[1][4];
        root_points[0][0] = cv::Point(0, 0);
        root_points[0][1] = cv::Point(rgb.cols, 0);
        root_points[0][2] = cv::Point(rgb.cols, corners[2].y);
        root_points[0][3] = cv::Point(0, corners[3].y);
        const cv::Point* ppt[1] = { root_points[0] };
        int npt[] = { 4 };
        polylines(rgb, ppt, npt, 1, 1, cv::Scalar(255), 1, 8, 0);
        fillPoly(rgb, ppt, npt, 1, cv::Scalar(0, 0, 0));

        cv::Point root_points2[1][4];
        root_points2[0][0] = cv::Point(0, corners[1].y);
        root_points2[0][1] = cv::Point(rgb.cols, corners[0].y);
        root_points2[0][2] = cv::Point(rgb.cols, rgb.rows);
        root_points2[0][3] = cv::Point(0, rgb.rows);
        const cv::Point* ppt2[1] = { root_points2[0] };
        int npt2[] = { 4 };
        polylines(rgb, ppt2, npt2, 1, 1, cv::Scalar(255), 1, 8, 0);
        fillPoly(rgb, ppt2, npt2, 1, cv::Scalar(0, 0, 0));

        std::vector<cv::Point> dd = get4point2(rgb);

        return dd;

    }

    std::vector<cv::Point> FineMapping::get4point2(cv::Mat InputProposal,
                                                   int sliceNum, int upper,
                                                   int lower, int windows_size) {

        cv::Mat proposal;

        cv::Mat c, c2, c3,c4,c5;
        InputProposal.copyTo(c);
        InputProposal.copyTo(c2);
        InputProposal.copyTo(c3);
        InputProposal.copyTo(c4);
        InputProposal.copyTo(c5);

        if (InputProposal.channels() == 3) {
            cv::cvtColor(InputProposal, proposal, cv::COLOR_BGR2GRAY);
        } else {
            InputProposal.copyTo(proposal);
        }

        float diff = static_cast<float>(upper - lower);
        diff /= static_cast<float>(sliceNum - 1);
        cv::Mat binary_adaptive;
        std::vector<cv::Point> line_upper;
        std::vector<cv::Point> line_lower;

        std::vector<cv::Point> line_upper_left_top;
        std::vector<cv::Point> line_lower_right_bottom;
        std::vector<cv::Point> line_upper_right_top;
        std::vector<cv::Point> line_lower_left_bottom;

        std::vector<std::pair<cv::Point, cv::Rect>> line_upper_left_top2;
        std::vector<std::pair<cv::Point, cv::Rect>> line_lower_right_bottom2;
        std::vector<std::pair<cv::Point, cv::Rect>> line_upper_right_top2;
        std::vector<std::pair<cv::Point, cv::Rect>> line_lower_left_bottom2;

        std::vector<std::pair<cv::Rect, int>> arr2;

        for (int i = 0; i < sliceNum; i++) {

            std::vector<std::vector<cv::Point>> contours;

            float k = lower + i*diff;

            //自适应二值化
            cv::adaptiveThreshold(proposal, binary_adaptive,
                                  255,
                                  cv::ADAPTIVE_THRESH_MEAN_C,
                                  cv::THRESH_BINARY,
                                  windows_size, k);

            //找轮廓
            cv::findContours(binary_adaptive, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            for (auto contour : contours) {

                //根据轮廓获取最小的矩形
                cv::Rect bdbox = cv::boundingRect(contour);

                //矩形的高/宽
                float lwRatio = bdbox.height / static_cast<float>(bdbox.width);
                //矩形的面积
                int  bdboxAera = bdbox.width*bdbox.height;

                if (lwRatio > 1 && bdboxAera > 100) {

                    cv::Point p1(bdbox.x, bdbox.y);//left-top
                    cv::Point p2(bdbox.x + bdbox.width, bdbox.y + bdbox.height);//right-bottom

                    line_upper_left_top.push_back(p1);
                    line_lower_right_bottom.push_back(p2);

                    cv::Point p3(bdbox.x + bdbox.width, bdbox.y);//right-top
                    cv::Point p4(bdbox.x, bdbox.y + bdbox.height);//left-bottom

                    line_upper_right_top.push_back(p3);
                    line_lower_left_bottom.push_back(p4);

                    std::pair<cv::Point, cv::Rect> s1;
                    s1.first = p1; s1.second = bdbox;
                    std::pair<cv::Point, cv::Rect> s2;
                    s2.first = p2; s2.second = bdbox;
                    std::pair<cv::Point, cv::Rect> s3;
                    s3.first = p3; s3.second = bdbox;
                    std::pair<cv::Point, cv::Rect> s4;
                    s4.first = p4; s4.second = bdbox;
                    line_upper_left_top2.push_back(s1);
                    line_lower_right_bottom2.push_back(s2);
                    line_upper_right_top2.push_back(s3);
                    line_lower_left_bottom2.push_back(s4);

                    int kk = 0, tag = 0;
                    for (auto item : arr2) {

                        if (item.first.width*item.first.height == bdboxAera) {
                            item.second += 1;
                            tag = 1;

                            arr2.erase(arr2.begin() + kk);

                            std::pair<cv::Rect, int> t;
                            t.first = bdbox;
                            t.second = item.second + 1;
                            arr2.push_back(t);

                            break;
                        }

                        kk++;

                    }
                    if (tag == 0) {
                        std::pair<cv::Rect, int> t;
                        t.first = bdbox;
                        t.second = 0;
                        arr2.push_back(t);
                    }

                }

            }

        }


        //获得准确的轮廓矩形个体
        cv::Rect rectOK = getArea2(arr2);

        //获得比较准确的点集合
        std::vector<cv::Point> line_upper_left_top3 = guolv2(line_upper_left_top2, rectOK);
        std::vector<cv::Point> line_lower_right_bottom3 = guolv2(line_lower_right_bottom2, rectOK);
        std::vector<cv::Point> line_upper_right_top3 = guolv2(line_upper_right_top2, rectOK);
        std::vector<cv::Point> line_lower_left_bottom3 = guolv2(line_lower_left_bottom2, rectOK);

        cv::Mat rgb;
        //给图片增加边距
        cv::copyMakeBorder(InputProposal,
                           rgb,
                           PADDING_UP_DOWN,
                           PADDING_UP_DOWN,
                           0, 0, cv::BORDER_REPLICATE);


        std::pair<int, int> A;
        std::pair<int, int> B;

        A = FitLineRansac2(line_upper_right_top3, rgb.cols, -1);
        B = FitLineRansac2(line_lower_left_bottom3, rgb.cols, 1);
        int leftyB = A.first;
        int rightyB = A.second;
        int leftyA = B.first;
        int rightyA = B.second;
        int cols = rgb.cols;
        int rows = rgb.rows;

        cv::Point2f p1 = cv::Point2f(0, A.first);
        cv::Point2f p2 = cv::Point2f(cols - 1, A.second);
        double dg = getXieLv(p1, p2);

        if (dg < 0) {
            A = FitLineRansac2(line_upper_left_top3, rgb.cols, -1);
            B = FitLineRansac2(line_lower_right_bottom3, rgb.cols, 1);
            leftyB = A.first;
            rightyB = A.second;
            leftyA = B.first;
            rightyA = B.second;
            cols = rgb.cols;
            rows = rgb.rows;
        }


        std::vector<cv::Point2f> corners(4);
        corners[0] = cv::Point2f(cols - 1, rightyA);//bottom-right
        corners[1] = cv::Point2f(0, leftyA);//bottom-left
        corners[2] = cv::Point2f(cols - 1, rightyB);//top-right
        corners[3] = cv::Point2f(0, leftyB);//top-left

        //获得比较准确的轮廓点矩形集合
        std::vector<std::pair<cv::Point, cv::Rect>> line_upper_left_top33 = guolv3(line_upper_left_top2, rectOK);
        cv::Rect will_cr = getWillCenterRect(line_upper_left_top33, InputProposal);
        cv::Point cp = getCenterPoint(InputProposal);

        //中心矩形
        cv::Rect cr = getCenterRect(will_cr, cp);
        std::vector<cv::Rect> list_rect(7);
        if (cr.x > 0) {
            //存在中心的矩形

            //根据中心矩形推出其它矩形一个一个的
            list_rect = getRects2(InputProposal, cr, line_upper_left_top33, dg);
        }

        cv::Point p_left_top(list_rect[0].x,list_rect[0].y);
        cv::Point p_left_bottom(list_rect[0].x, list_rect[0].y+list_rect[0].height);
        cv::Point p_right_top(list_rect[6].x+list_rect[6].width, list_rect[6].y);
        cv::Point p_right_bottom(list_rect[6].x+list_rect[6].width, list_rect[6].y+list_rect[6].height);

        std::vector<cv::Point> corners_trans(4);
        corners_trans[0] = p_left_top;
        corners_trans[1] = p_right_top;
        corners_trans[2] = p_right_bottom;
        corners_trans[3] = p_left_bottom;

        return corners_trans;

    }




    std::vector<cv::Rect> get7Rect2(cv::Mat InputProposal,
                                    int sliceNum, int upper, int lower,
                                    int windows_size) {

        cv::Mat proposal;

        cv::Mat c, c2, c3, c4, c5;
        InputProposal.copyTo(c);
        InputProposal.copyTo(c2);
        InputProposal.copyTo(c3);
        InputProposal.copyTo(c4);
        InputProposal.copyTo(c5);

        if (InputProposal.channels() == 3) {
            cv::cvtColor(InputProposal, proposal, cv::COLOR_BGR2GRAY);
        }
        else {
            InputProposal.copyTo(proposal);
        }

        float diff = static_cast<float>(upper - lower);
        diff /= static_cast<float>(sliceNum - 1);
        cv::Mat binary_adaptive;
        std::vector<cv::Point> line_upper;
        std::vector<cv::Point> line_lower;

        std::vector<cv::Point> line_upper_left_top;
        std::vector<cv::Point> line_lower_right_bottom;
        std::vector<cv::Point> line_upper_right_top;
        std::vector<cv::Point> line_lower_left_bottom;


        std::vector<std::pair<cv::Point, cv::Rect>> line_upper_left_top2;
        std::vector<std::pair<cv::Point, cv::Rect>> line_lower_right_bottom2;
        std::vector<std::pair<cv::Point, cv::Rect>> line_upper_right_top2;
        std::vector<std::pair<cv::Point, cv::Rect>> line_lower_left_bottom2;

        std::vector<std::pair<cv::Rect, int>> arr2;

        for (int i = 0; i < sliceNum; i++)
        {

            std::vector<std::vector<cv::Point>> contours;

            float k = lower + i*diff;

            //自适应二值化
            cv::adaptiveThreshold(proposal, binary_adaptive,
                                  255,
                                  cv::ADAPTIVE_THRESH_MEAN_C,
                                  cv::THRESH_BINARY,
                                  windows_size, k);

            //找轮廓
            cv::findContours(binary_adaptive, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            for (auto contour : contours)
            {
                //根据轮廓获取最小的矩形
                cv::Rect bdbox = cv::boundingRect(contour);

                //矩形的高/宽
                float lwRatio = bdbox.height / static_cast<float>(bdbox.width);
                //矩形的面积
                int  bdboxAera = bdbox.width*bdbox.height;


                if (lwRatio > 1 && bdboxAera > 100)
                {
                    cv::Point p1(bdbox.x, bdbox.y);//left-top
                    cv::Point p2(bdbox.x + bdbox.width, bdbox.y + bdbox.height);//right-bottom

                    line_upper_left_top.push_back(p1);
                    line_lower_right_bottom.push_back(p2);

                    cv::Point p3(bdbox.x + bdbox.width, bdbox.y);//right-top
                    cv::Point p4(bdbox.x, bdbox.y + bdbox.height);//left-bottom

                    line_upper_right_top.push_back(p3);
                    line_lower_left_bottom.push_back(p4);



                    std::pair<cv::Point, cv::Rect> s1;
                    s1.first = p1; s1.second = bdbox;
                    std::pair<cv::Point, cv::Rect> s2;
                    s2.first = p2; s2.second = bdbox;
                    std::pair<cv::Point, cv::Rect> s3;
                    s3.first = p3; s3.second = bdbox;
                    std::pair<cv::Point, cv::Rect> s4;
                    s4.first = p4; s4.second = bdbox;
                    line_upper_left_top2.push_back(s1);
                    line_lower_right_bottom2.push_back(s2);
                    line_upper_right_top2.push_back(s3);
                    line_lower_left_bottom2.push_back(s4);

                    int kk = 0, tag = 0;
                    for (auto item : arr2) {

                        if (item.first.width*item.first.height == bdboxAera) {
                            item.second += 1;
                            tag = 1;

                            arr2.erase(arr2.begin() + kk);

                            std::pair<cv::Rect, int> t;
                            t.first = bdbox;
                            t.second = item.second + 1;
                            arr2.push_back(t);

                            break;
                        }

                        kk++;

                    }
                    if (tag == 0) {
                        std::pair<cv::Rect, int> t;
                        t.first = bdbox;
                        t.second = 0;
                        arr2.push_back(t);
                    }


                }


            }

        }


        //获得准确的轮廓矩形个体
        cv::Rect rectOK = getArea2(arr2);

        //获得比较准确的点集合
        std::vector<cv::Point> line_upper_left_top3 = guolv2(line_upper_left_top2, rectOK);
        std::vector<cv::Point> line_lower_right_bottom3 = guolv2(line_lower_right_bottom2, rectOK);
        std::vector<cv::Point> line_upper_right_top3 = guolv2(line_upper_right_top2, rectOK);
        std::vector<cv::Point> line_lower_left_bottom3 = guolv2(line_lower_left_bottom2, rectOK);

        cv::Mat rgb;
        //给图片增加边距
        cv::copyMakeBorder(InputProposal,
                           rgb,
                           PADDING_UP_DOWN,
                           PADDING_UP_DOWN,
                           0, 0, cv::BORDER_REPLICATE);


        std::pair<int, int> A;
        std::pair<int, int> B;

        A = FitLineRansac2(line_upper_right_top3, rgb.cols, -1);
        B = FitLineRansac2(line_lower_left_bottom3, rgb.cols, 1);
        int leftyB = A.first;
        int rightyB = A.second;
        int leftyA = B.first;
        int rightyA = B.second;
        int cols = rgb.cols;
        int rows = rgb.rows;

        cv::Point2f p1 = cv::Point2f(0, A.first);
        cv::Point2f p2 = cv::Point2f(cols - 1, A.second);
        double dg = getXieLv(p1, p2);

        if (dg < 0) {
            A = FitLineRansac2(line_upper_left_top3, rgb.cols, -1);
            B = FitLineRansac2(line_lower_right_bottom3, rgb.cols, 1);
            leftyB = A.first;
            rightyB = A.second;
            leftyA = B.first;
            rightyA = B.second;
            cols = rgb.cols;
            rows = rgb.rows;
        }


        std::vector<cv::Point2f> corners(4);
        corners[0] = cv::Point2f(cols - 1, rightyA);//bottom-right
        corners[1] = cv::Point2f(0, leftyA);//bottom-left
        corners[2] = cv::Point2f(cols - 1, rightyB);//top-right
        corners[3] = cv::Point2f(0, leftyB);//top-left




        //获得比较准确的轮廓点矩形集合
        std::vector<std::pair<cv::Point, cv::Rect>> line_upper_left_top33 = guolv3(line_upper_left_top2, rectOK);
        line_upper_left_top33 = guolv3_2(line_upper_left_top33, rectOK);


        //中心矩形
        cv::Rect cr ;
        cr = getCenterRect(line_upper_left_top33, InputProposal,rectOK,dg);

        std::vector<cv::Rect> list_rect(7);
        std::pair<int, std::vector<cv::Rect>> pair;

        if (cr.area() > 0) {
            //存在中心的矩形

            //pair = getRects3(InputProposal, cr, line_upper_left_top33, dg);
            //pair = getRects4(InputProposal, cr, line_upper_left_top33, dg);

            pair = getRects5(rgb, cr, line_upper_left_top33, dg);
            if (pair.first == 1) {
                list_rect = pair.second;
            }

        }

        return list_rect;

    }

    std::vector<cv::Rect> FineMapping::get7Rect(cv::Mat InputProposal,
                                                int sliceNum, int upper,
                                                int lower, int windows_size) {

        cv::Mat proposal;

        cv::Mat c, c2, c3;
        InputProposal.copyTo(c);
        InputProposal.copyTo(c2);
        InputProposal.copyTo(c3);

        if (InputProposal.channels() == 3) {
            cv::cvtColor(InputProposal, proposal, cv::COLOR_BGR2GRAY);
        }
        else {
            InputProposal.copyTo(proposal);
        }

        float diff = static_cast<float>(upper - lower);
        diff /= static_cast<float>(sliceNum - 1);
        cv::Mat binary_adaptive;
        std::vector<cv::Point> line_upper;
        std::vector<cv::Point> line_lower;

        std::vector<cv::Point> line_upper_left_top;
        std::vector<cv::Point> line_lower_right_bottom;
        std::vector<cv::Point> line_upper_right_top;
        std::vector<cv::Point> line_lower_left_bottom;

        for (int i = 0; i < sliceNum; i++)
        {

            std::vector<std::vector<cv::Point>> contours;

            float k = lower + i*diff;

            //自适应二值化
            cv::adaptiveThreshold(proposal, binary_adaptive,
                                  255,
                                  cv::ADAPTIVE_THRESH_MEAN_C,
                                  cv::THRESH_BINARY,
                                  windows_size, k);

            //找轮廓
            cv::findContours(binary_adaptive, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            int cc = 0;
            for (auto contour : contours)
            {
                //根据轮廓获取最小的矩形
                cv::Rect bdbox = cv::boundingRect(contour);

                //矩形的高/宽
                float lwRatio = bdbox.height / static_cast<float>(bdbox.width);
                //矩形的面积
                int  bdboxAera = bdbox.width*bdbox.height;

                if (lwRatio > 1 && bdboxAera > 100)
                {
                    cv::Point p1(bdbox.x, bdbox.y - 5);//left-top
                    cv::Point p2(bdbox.x + bdbox.width, bdbox.y + bdbox.height + 5);//right-bottom

                    line_upper_left_top.push_back(p1);
                    line_lower_right_bottom.push_back(p2);

                    cv::Point p3(bdbox.x + bdbox.width, bdbox.y - 5);//right-top
                    cv::Point p4(bdbox.x, bdbox.y + bdbox.height + 5);//left-bottom

                    line_upper_right_top.push_back(p3);
                    line_lower_left_bottom.push_back(p4);

                }


            }

        }

        cv::Mat rgb;
        //给图片增加边距
        cv::copyMakeBorder(InputProposal,
                           rgb,
                           PADDING_UP_DOWN,
                           PADDING_UP_DOWN,
                           0, 0, cv::BORDER_REPLICATE);

        std::pair<int, int> A;
        std::pair<int, int> B;

        A = FitLineRansac2(line_upper_right_top, rgb.cols, -1);
        B = FitLineRansac2(line_lower_left_bottom, rgb.cols, 1);
        int leftyB = A.first;
        int rightyB = A.second;
        int leftyA = B.first;
        int rightyA = B.second;
        int cols = rgb.cols;
        int rows = rgb.rows;


        cv::Point2f p1 = cv::Point2f(0, A.first);
        cv::Point2f p2 = cv::Point2f(cols - 1, A.second);
        double dg = getXieLv(p1, p2);

        if (dg < 0) {
            A = FitLineRansac2(line_upper_left_top, rgb.cols, -1);
            B = FitLineRansac2(line_lower_right_bottom, rgb.cols, 1);
            leftyB = A.first;
            rightyB = A.second;
            leftyA = B.first;
            rightyA = B.second;
            cols = rgb.cols;
            rows = rgb.rows;
        }


        std::vector<cv::Point2f> corners(4);
        corners[0] = cv::Point2f(cols - 1, rightyA);//bottom-right
        corners[1] = cv::Point2f(0, leftyA);//bottom-left
        corners[2] = cv::Point2f(cols - 1, rightyB);//top-right
        corners[3] = cv::Point2f(0, leftyB);//top-left



        //遮去无效部分
        cv::Point root_points[1][4];
        root_points[0][0] = cv::Point(0, 0);
        root_points[0][1] = cv::Point(rgb.cols, 0);
        root_points[0][2] = cv::Point(rgb.cols, corners[2].y);
        root_points[0][3] = cv::Point(0, corners[3].y);
        const cv::Point* ppt[1] = { root_points[0] };
        int npt[] = { 4 };
        polylines(rgb, ppt, npt, 1, 1, cv::Scalar(255), 1, 8, 0);
        fillPoly(rgb, ppt, npt, 1, cv::Scalar(0, 0, 0));

        cv::Point root_points2[1][4];
        root_points2[0][0] = cv::Point(0, corners[1].y);
        root_points2[0][1] = cv::Point(rgb.cols, corners[0].y);
        root_points2[0][2] = cv::Point(rgb.cols, rgb.rows);
        root_points2[0][3] = cv::Point(0, rgb.rows);
        const cv::Point* ppt2[1] = { root_points2[0] };
        int npt2[] = { 4 };
        polylines(rgb, ppt2, npt2, 1, 1, cv::Scalar(255), 1, 8, 0);
        fillPoly(rgb, ppt2, npt2, 1, cv::Scalar(0, 0, 0));

        std::vector<cv::Rect> dd = get7Rect2(rgb, 15, 0, -50, 17);

        return dd;

    }



}


